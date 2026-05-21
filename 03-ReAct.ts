import { ChatOpenAI } from "@langchain/openai";
import { TavilySearch } from "@langchain/tavily";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StateGraph, END, MessagesAnnotation } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { z } from "zod";

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
const searchTool = new TavilySearch({ maxResults: 2, name: "web_search", tavilyApiKey: process.env.TAVILY_API_KEY, });

const llm = new ChatOpenAI({
  modelName: "deepseek-ai/DeepSeek-V4-Pro", // 这里替换为魔搭支持的模型
  temperature: 0.2,
});

const llmWithTools = llm.bindTools([searchTool]);
async function reactAgentNode(state: typeof MessagesAnnotation.State) {
  console.log("--- REACT 智能体：正在思考... ---");
  const response = await llmWithTools.invoke(state.messages);
  return { messages: [response] };
}

const reactToolNode = new ToolNode([searchTool]);

// 我们使用内置的 toolsCondition 作为路由，逻辑与之前相同：
// 如果最后一条消息有 tool_calls，则走向 "tools"；否则走向 END。

// 现在我们定义带有关键循环的图
const reactGraphBuilder = new StateGraph(MessagesAnnotation)
  .addNode("agent", reactAgentNode)
  .addNode("tools", reactToolNode)
  .addEdge("__start__", "agent")
  .addConditionalEdges("agent", toolsCondition)
  // 这是关键的区别：边从 tools 返回到 agent
  .addEdge("tools", "agent");

const reactAgentApp = reactGraphBuilder.compile();
console.log("ReAct 智能体编译成功，已包含推理循环。");

const multiStepQuery = "制作科幻电影《沙丘》的公司的现任CEO是谁？该公司最新一部电影的预算是多少？";

console.log(`\n[测试 ReAct 智能体处理相同的多步查询]：'${multiStepQuery}'\n`);

let finalReactOutput: any = null;
const stream = await reactAgentApp.stream(
  { messages: [new HumanMessage(multiStepQuery)] },
  { streamMode: "values" }
);

for await (const chunk of stream) {
  finalReactOutput = chunk;
  console.log("--- 当前状态 ---");
  const lastMsg = chunk.messages[chunk.messages.length - 1];
  console.log(`[${lastMsg!.type}]`, lastMsg!.content || "(Tool Call / Empty Content)");
  if (lastMsg!.tool_calls && lastMsg!.tool_calls.length > 0) {
    console.log("工具调用:", JSON.stringify(lastMsg!.tool_calls, null, 2));
  }
  console.log("\n");
}

console.log("\n--- ReAct 智能体的最终输出 ---");
const finalReactMessage = finalReactOutput.messages[finalReactOutput.messages.length - 1];
console.log(finalReactMessage.content);

// 定义评估的 Zod Schema
const TaskEvaluation = z.object({
  task_completion_score: z.number().int().describe("1-10分，评估智能体是否成功完成了用户请求的所有部分。"),
  reasoning_quality_score: z.number().int().describe("1-10分，评估智能体展示的逻辑流和推理过程。"),
  justification: z.string().describe("给出评分的简短理由。")
});

// 使用结构化输出绑定 LLM
const judgeLlm = llm.withStructuredOutput(TaskEvaluation);

async function evaluateAgentOutput(query: string, agentOutput: any) {
  const trace = agentOutput.messages
    .map((m: any) => {
      let content = m.content;
      if (m.tool_calls && m.tool_calls.length > 0) {
        content += `\nTool Calls: ${JSON.stringify(m.tool_calls)}`;
      }
      return `${m._getType()}: ${content}`;
    })
    .join("\n");

  const prompt = `你是一个评估 AI 智能体的专家裁判。请在 1-10 分的范围内评估以下智能体在给定任务上的表现。10 分表示任务完美完成，1 分表示彻底失败。
  
  **用户的任务：**
  ${query}
  
  **完整的智能体对话追踪：**
  \`\`\`
  ${trace}
  \`\`\`
  `;

  return await judgeLlm.invoke(prompt);
}

// 定义基础智能体的节点
async function basicAgentNode(state: typeof MessagesAnnotation.State) {
  console.log("--- 基础智能体：正在思考... ---");
  // 注意：我们提供了一个系统提示词，鼓励它在一次工具调用后直接回答
  const systemPrompt = new SystemMessage(
    "你是一个有用的助手。你可以使用网络搜索工具。请根据工具的结果回答用户的问题。你必须在一次工具调用后提供最终答案。"
  );
  
  const messages = [systemPrompt, ...state.messages];
  const response = await llmWithTools.invoke(messages);
  
  return { messages: [response] };
}

// 定义基础的线性图
const basicGraphBuilder = new StateGraph(MessagesAnnotation)
  .addNode("agent", basicAgentNode)
  .addNode("tools", new ToolNode([searchTool]))
  .addEdge("__start__", "agent")
  // 智能体执行后，只能走向 tools，或者结束。
  .addConditionalEdges("agent", toolsCondition)
  // tools 执行后，必须结束（没有循环）
  .addEdge("tools", END);

const basicToolAgentApp = basicGraphBuilder.compile();

console.log("基础单次工具调用智能体编译成功。");

const basicAgentOutput = await basicToolAgentApp.invoke({
  messages: [new HumanMessage(multiStepQuery)]
});

console.log("--- 评估基础智能体的输出 ---");
const basicAgentEvaluation = await evaluateAgentOutput(multiStepQuery, basicAgentOutput);
console.log(JSON.stringify(basicAgentEvaluation, null, 2));

console.log("\n--- 评估 ReAct 智能体的输出 ---");
const reactAgentEvaluation = await evaluateAgentOutput(multiStepQuery, finalReactOutput);
console.log(JSON.stringify(reactAgentEvaluation, null, 2));