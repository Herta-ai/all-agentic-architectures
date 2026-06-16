import { ChatOpenAI } from "@langchain/openai";
import { TavilySearch } from "@langchain/tavily";
import { AIMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StateGraph, START, END, MessagesAnnotation } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { z } from "zod";
import chalk from "chalk";

// --- API 密钥和 Tracing 设置 ---
// Bun 会自动加载 .env 文件，但我们需要设置 LangSmith 的环境变量
process.env.LANGCHAIN_PROJECT = "Agentic Architecture - ReAct";

const requiredKeys = ["OPENAI_BASE_URL", "OPENAI_API_KEY", "LANGSMITH_TRACING", "TAVILY_API_KEY"];
for (const key of requiredKeys) {
  if (!process.env[key]) {
    console.warn(`未找到 ${key}。请在 .env 文件中进行设置。`);
  }
}

console.log("环境变量已加载，Tracing 设置完毕。");

// 定义工具和 LLM
const searchTool = new TavilySearch({
  maxResults: 2,
  name: 'web_search',
  description: '使用 Tavily 执行网络搜索并将结果作为字符串返回。',
  tavilyApiKey: process.env.TAVILY_API_KEY,
})

const llm = new ChatOpenAI({
  model: 'deepseek-v4-flash', // 这里替换为魔搭支持的模型
  temperature: 0.2,
})

const llmWithTools = llm.bindTools([searchTool])


// Agent 节点，带有一个系统提示词来强制一次仅调用一个工具
const reactAgentNode = async (state: typeof MessagesAnnotation.State) => {
  console.log(chalk.cyan("--- 反应式代理 (REACT): 思考中... ---"));
  
  const systemPrompt = new SystemMessage(
    "你是一个有用的研究助手。你每次必须只调用一个工具。不要在一次对话中调用多个工具。在收到工具结果后，你再决定下一步。"
  );

  const response = await llmWithTools.invoke([systemPrompt, ...state.messages]);
  return { messages: [response] };
};

// 将我们修正后的自定义工具放入 ToolNode
const toolNode = new ToolNode([searchTool]);

// 构建带有典型循环特征的 ReAct 图
const reactGraphBuilder = new StateGraph(MessagesAnnotation)
  .addNode("agent", reactAgentNode)
  .addNode("tools", toolNode)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", toolsCondition)
  .addEdge("tools", "agent");

const reactAgentApp = reactGraphBuilder.compile();
console.log(chalk.green("反应式 (ReAct) 代理已成功编译。"));

const planCentricQuery = `
查找法国、德国和意大利首都的人口。
然后计算它们的总人口。
最后，将该总人口与美国的人口进行比较，说明哪个更大。
`;

console.log(chalk.bold.yellow(`\n[在侧重计划的查询上测试反应式 Agent]: '${planCentricQuery}'\n`));

let finalReactOutput: typeof MessagesAnnotation.State | undefined;

// 使用 values 模式进行流式传输，以观察每一步的状态更新
const stream = await reactAgentApp.stream(
  { messages: [["user", planCentricQuery]] },
  { streamMode: "values" }
);

for await (const chunk of stream) {
  finalReactOutput = chunk;
  console.log(chalk.bold.magenta("--- 当前状态更新 ---"));
  const lastMessage = chunk.messages[chunk.messages.length - 1]!;
  console.log(`[${lastMessage.type}] ${lastMessage.content}`);
  if (AIMessage.isInstance(lastMessage) && lastMessage.tool_calls?.length) {
    console.log(`Tool Calls:`, lastMessage.tool_calls);
  }
  console.log("\n");
}

console.log(chalk.bold.red("\n--- 反应式 Agent 的最终输出 ---"));
if (finalReactOutput) {
  const finalMessage = finalReactOutput.messages[finalReactOutput.messages.length - 1]!;
  console.log(finalMessage.content);
}