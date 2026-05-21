# 📘 Agentic 架构 4：计划 (Planning)

在本笔记本中，我们将探索**计划 (Planning)** 架构。这种模式为 Agent (智能体) 的推理过程引入了至关重要的前瞻性层。与 ReAct 模型中一步一步对信息做出反应不同，计划型 Agent 会首先将一个复杂的任务分解为一系列较小的、易于管理的子目标。它会在采取任何实际行动*之前*，创建一个完整的“作战计划”。

这种积极主动的方法为多步任务带来了结构性、可预测性和高效率。为了突出它的优势，我们将直接比较**反应式 Agent (ReAct)** 和我们的**计划型 Agent (Planning)** 的性能。我们会向它们提供同一个需要收集多项信息然后执行最终计算的任务，从而展示预先计算好的计划如何带来更稳健、更直接的解决方案。

### 定义
**计划 (Planning)** 架构涉及一个 Agent，它在开始执行*之前*明确地将复杂目标分解为详细的子任务序列。这个初始计划阶段的输出是一个具体的、循序渐进的计划，然后 Agent 会有条不紊地遵循这个计划来达成解决方案。

### 高层工作流

1.  **接收目标：** Agent 接收到一个复杂的任务。
2.  **计划：** 一个专用的“计划员 (Planner)”组件分析目标，并生成实现目标所需子任务的有序列表。例如：`["找到事实A", "找到事实B", "使用A和B计算C"]`。
3.  **执行：** 一个“执行器 (Executor)”组件接收计划，并按顺序执行每个子任务，根据需要使用工具。
4.  **合成：** 一旦计划中的所有步骤都完成，最后一个组件会将执行步骤中收集到的结果合成为一个连贯的最终答案。

### 何时使用 / 应用场景
*   **多步工作流：** 非常适合操作顺序已知且关键的任务，例如生成需要获取数据、处理数据然后总结的报告。
*   **项目管理助手：** 将“推出新功能”等大型目标分解为不同团队的子任务。
*   **教育辅导：** 创建课程计划，教学生一个特定的概念，从基础原理到高级应用。

### 优点和缺点
*   **优点：**
    *   **结构化和可追踪：** 整个工作流程提前规划好，使得 Agent 的过程透明且易于调试。
    *   **高效：** 对于可预测的任务，比 ReAct 更高效，因为它避免了步骤之间不必要的推理循环。
*   **缺点：**
    *   **对变化脆弱：** 如果执行期间环境发生意外变化，预先制定的计划可能会失败。它的适应性不如可以在每一步后改变主意的 ReAct Agent。

## 第 0 阶段：基础和设置

我们将从标准设置过程开始：安装依赖并为 Nebius、LangSmith 和我们的 Tavily 网络搜索工具配置 API 密钥。

### 步骤 0.1：安装核心依赖 (Bun)

**我们将要做什么：**
我们使用 Bun 安装所需的库，包括 `@langchain/core`，`@langchain/langgraph`，`@langchain/openai` (用于连接兼容 OpenAI API 的 Nebius)，以及相关的工具包。

在终端中运行以下命令：
```bash
bun add @langchain/core @langchain/langgraph @langchain/openai @langchain/community zod dotenv chalk
```

### 步骤 0.2：导入库和设置密钥

**我们将要做什么：**
我们将导入必要的模块并从 `.env` 文件加载我们的 API 密钥。

**需要采取的行动：** 在当前目录下创建一个 `.env` 文件，包含以下密钥：
```env
OPENAI_BASE_URL=https://api-inference.modelscope.cn/v1
OPENAI_API_KEY="YOUR_NEBIUS_API"
LANGSMITH_TRACING=true
LANGSMITH_API_KEY="your_langsmith_api_key_here" 
TAVILY_API_KEY="YOUR_TAVILY_API"
LANGCHAIN_PROJECT="Agentic Architecture - Planning"
```

```typescript
import { z } from "zod";
import * as dotenv from "dotenv";
import chalk from "chalk";

// LangChain 组件
import { ChatOpenAI } from "@langchain/openai";
import { SystemMessage, ToolMessage, BaseMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";

// LangGraph 组件
import { 
  StateGraph, 
  MessagesAnnotation, 
  Annotation, 
  START, 
  END 
} from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";

// --- API 密钥和追踪设置 ---
dotenv.config();

// 检查密钥是否设置
const requiredKeys = ["NEBIUS_API_KEY", "LANGCHAIN_API_KEY", "TAVILY_API_KEY"];
for (const key of requiredKeys) {
  if (!process.env[key]) {
    console.warn(chalk.red(`${key} 未找到。请创建 .env 文件并设置它。`));
  }
}

console.log(chalk.green("环境变量已加载，LangSmith 追踪已设置。"));
```

## 第 1 阶段：基线 - 反应式 Agent (ReAct)

为了体会“计划”的价值，我们首先需要一个基线。我们将使用我们在之前的笔记本中构建的 ReAct Agent。这个 Agent 很智能，但有些短视——它一次只弄清楚一步的路径。

### 步骤 1.1：重建 ReAct Agent

**我们将要做什么：**
我们将快速重建 ReAct Agent。它的核心特征是一个循环，Agent 在每次工具调用后，其输出都会被路由回自身，从而使其能够根据最新信息重新评估并决定下一步行动。

```typescript
// 1. 定义基础的 Tavily 搜索工具
const tavilySearchTool = new TavilySearchResults({ maxResults: 2 });

// 2. 自定义工具：使用 @langchain/core/tools 的 `tool` 函数封装它
const webSearch = tool(
  async ({ query }) => {
    console.log(chalk.blue(`--- 工具: 正在搜索 '${query}'...`));
    const results = await tavilySearchTool.invoke(query);
    // 返回字符串格式以保持与原始 Python 代码类似的行为
    return typeof results === 'string' ? results : JSON.stringify(results);
  },
  {
    name: "web_search",
    description: "使用 Tavily 执行网络搜索并将结果作为字符串返回。",
    schema: z.object({ query: z.string() }),
  }
);

// 3. 定义 LLM 并将其与我们的自定义工具绑定
// Nebius AI 兼容 OpenAI 的 API 格式，所以我们使用 ChatOpenAI 并指定 baseURL
const llm = new ChatOpenAI({
  modelName: "meta-llama/Meta-Llama-3.1-8B-Instruct",
  temperature: 0,
  openAIApiKey: process.env.NEBIUS_API_KEY,
  configuration: {
    baseURL: "https://api.studio.nebius.ai/v1/",
  },
});

const llmWithTools = llm.bindTools([webSearch]);

// 4. Agent 节点，带有一个系统提示词来强制一次仅调用一个工具
const reactAgentNode = async (state: typeof MessagesAnnotation.State) => {
  console.log(chalk.cyan("--- 反应式代理 (REACT): 思考中... ---"));
  
  const systemPrompt = new SystemMessage(
    "你是一个有用的研究助手。你每次必须只调用一个工具。不要在一次对话中调用多个工具。在收到工具结果后，你再决定下一步。"
  );

  const response = await llmWithTools.invoke([systemPrompt, ...state.messages]);
  return { messages: [response] };
};

// 5. 将我们修正后的自定义工具放入 ToolNode
const toolNode = new ToolNode([webSearch]);

// 构建带有典型循环特征的 ReAct 图
const reactGraphBuilder = new StateGraph(MessagesAnnotation)
  .addNode("agent", reactAgentNode)
  .addNode("tools", toolNode)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", toolsCondition)
  .addEdge("tools", "agent");

const reactAgentApp = reactGraphBuilder.compile();
console.log(chalk.green("反应式 (ReAct) 代理已成功编译。"));
```

### 步骤 1.2：在侧重计划的问题上测试反应式 Agent

**我们将要做什么：**
我们将给 ReAct Agent 分配一个任务，该任务需要两个独立的数据收集步骤，然后进行最终计算。这将测试其在没有前期计划的情况下管理多步工作流的能力。

```typescript
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
  const lastMessage = chunk.messages[chunk.messages.length - 1];
  console.log(`[${lastMessage._getType()}] ${lastMessage.content}`);
  if (lastMessage.tool_calls?.length) {
    console.log(`Tool Calls:`, lastMessage.tool_calls);
  }
  console.log("\n");
}

console.log(chalk.bold.red("\n--- 反应式 Agent 的最终输出 ---"));
if (finalReactOutput) {
  const finalMessage = finalReactOutput.messages[finalReactOutput.messages.length - 1];
  console.log(finalMessage.content);
}
```

**对输出的讨论：**
ReAct Agent 成功完成了任务。通过观察流式输出，我们可以追踪其一步步的推理过程：
1. 它首先决定搜索巴黎的人口。
2. 收到该结果后，它将其整合到记忆中，然后决定下一步是搜索柏林的人口。
3. 最后，在收集到所有信息后，它执行了计算并提供了最终答案。

虽然它有效，但这种迭代发现的过程并不总是最有效的。对于像这样可预测的任务，Agent 在每一步之间进行了额外的 LLM 调用来进行推理。这就为展示计划型 Agent 的价值奠定了基础。

## 第 2 阶段：高级方法 - 计划型 Agent (Planning Agent)

现在，让我们构建一个“三思而后行”的 Agent。这个 Agent 将拥有一个专用的**计划员 (Planner)** 来创建完整的任务列表，一个**执行器 (Executor)** 来执行计划，以及一个**合成器 (Synthesizer)** 来组合最终结果。

### 步骤 2.1：定义计划员、执行器和合成器节点

**我们将要做什么：**
我们将为我们的新 Agent 创建核心组件：
1.  **`Planner`:** 基于 LLM 的节点，接收用户请求并输出结构化计划。
2.  **`Executor`:** 接收计划，使用工具执行*下一个*步骤，并记录结果的节点。
3.  **`Synthesizer`:** 最终的基于 LLM 的节点，接收所有收集到的结果并生成最终答案。

```typescript
// 1. 使用 Zod 定义确保计划员输出是结构化步骤列表的模式
const PlanSchema = z.object({
  steps: z.array(z.string()).describe("一系列工具调用的列表。一旦全部执行完毕，将能回答查询。"),
});

// 2. 为计划型 Agent 定义图状态 (State)
const PlanningStateAnnotation = Annotation.Root({
  user_request: Annotation<string>,
  plan: Annotation<string[]>({
    reducer: (state, update) => update, // 覆盖更新计划数组
  }),
  intermediate_steps: Annotation<ToolMessage[]>({
    reducer: (state, update) => state.concat(update), // 累加中间步骤
    default: () => [],
  }),
  final_answer: Annotation<string>,
});

// 3. 计划员节点
const plannerNode = async (state: typeof PlanningStateAnnotation.State) => {
  console.log(chalk.cyan("--- 计划员 (PLANNER): 正在分解任务... ---"));
  const plannerLlm = llm.withStructuredOutput(PlanSchema);
  
  // 带有明确示例 (Few-shot prompting) 的提示词
  const prompt = `你是一个专家级的计划员。你的工作是创建一个循序渐进的计划来回答用户的请求。
计划中的每一步都必须是对 \`web_search\` 工具的一次单独调用。

**指示：**
1. 分析用户的请求。
2. 将其分解为一系列简单的、合乎逻辑的搜索查询。
3. 输出格式为字符串数组，其中每个字符串都是一次有效的工具调用。

**示例：**
请求: "法国的首都及其人口是多少？"
正确的计划输出 (JSON 格式):
{
  "steps": [
    "web_search('法国的首都')",
    "web_search('巴黎的人口')"
  ]
}

**用户的请求：**
${state.user_request}`;

  const planResult = await plannerLlm.invoke(prompt);
  console.log(chalk.cyan(`--- 计划员 (PLANNER): 生成的计划: [${planResult.steps.join(", ")}] ---`));
  
  return { plan: planResult.steps };
};

// 4. 执行器节点
const executorNode = async (state: typeof PlanningStateAnnotation.State) => {
  console.log(chalk.cyan("--- 执行器 (EXECUTOR): 运行下一步... ---"));
  const plan = state.plan;
  const nextStep = plan[0];
  
  // 使用健壮的正则来处理单引号和双引号
  const match = nextStep.match(/(\w+)\((?:"|')(.*?)(?:"|')\)/);
  let toolName = "web_search";
  let query = nextStep;
  
  if (match) {
    toolName = match[1];
    query = match[2];
  }
  
  console.log(chalk.blue(`--- 执行器 (EXECUTOR): 调用工具 '${toolName}'，查询 '${query}' ---`));
  
  const result = await tavilySearchTool.invoke(query);
  
  // 我们创建 ToolMessage，现在的工具调用是非常安全的。
  const toolMessage = new ToolMessage({
    content: typeof result === 'string' ? result : JSON.stringify(result),
    name: toolName,
    tool_call_id: `manual-${Date.now().toString()}`,
  });
  
  return {
    plan: plan.slice(1), // 将已经执行的步骤从计划中移除
    intermediate_steps: [toolMessage], // reducer 会将其拼接到数组末尾
  };
};

// 5. 合成器节点
const synthesizerNode = async (state: typeof PlanningStateAnnotation.State) => {
  console.log(chalk.cyan("--- 合成器 (SYNTHESIZER): 生成最终答案... ---"));
  
  const context = state.intermediate_steps
    .map((msg) => `工具 ${msg.name} 返回: ${msg.content}`)
    .join("\n");
  
  const prompt = `你是一个专家级的合成器。根据用户的请求和收集到的数据，提供一个全面的最终答案。
  
请求: ${state.user_request}
收集到的数据:
${context}`;

  const finalAnswer = await llm.invoke(prompt);
  return { final_answer: finalAnswer.content as string };
};

console.log(chalk.green("计划员、执行器和合成器节点已定义完毕。"));
```

### 步骤 2.2：构建计划型 Agent 图 (Graph)

**我们将要做什么：**
现在我们将把这些新节点组装成一个图。工作流将是：`Planner` -> `Executor` (循环) -> `Synthesizer`。

```typescript
const planningRouter = (state: typeof PlanningStateAnnotation.State) => {
  if (!state.plan || state.plan.length === 0) {
    console.log(chalk.yellow("--- 路由器 (ROUTER): 计划已完成。转到合成器。 ---"));
    return "synthesize";
  } else {
    console.log(chalk.yellow("--- 路由器 (ROUTER): 计划还有更多步骤。继续执行。 ---"));
    return "execute";
  }
};

const planningGraphBuilder = new StateGraph(PlanningStateAnnotation)
  .addNode("plan", plannerNode)
  .addNode("execute", executorNode)
  .addNode("synthesize", synthesizerNode)
  .addEdge(START, "plan")
  // 计划之后进行路由判断
  .addConditionalEdges("plan", planningRouter)
  // 执行之后再次进行路由判断
  .addConditionalEdges("execute", planningRouter)
  .addEdge("synthesize", END);

const planningAgentApp = planningGraphBuilder.compile();
console.log(chalk.green("计划型 Agent 已成功编译。"));
```

## 第 3 阶段：正面交锋的比较

让我们在同一个任务上运行我们全新的计划型 Agent，并将其执行流程和最终输出与反应式 Agent 进行比较。

```typescript
console.log(chalk.bold.green(`\n[在同一个查询上测试计划型 Agent]: '${planCentricQuery}'\n`));

// 初始状态输入
const initialPlanningInput = {
  user_request: planCentricQuery,
};

const finalPlanningOutput = await planningAgentApp.invoke(initialPlanningInput);

console.log(chalk.bold.green("\n--- 计划型 Agent 的最终输出 ---"));
console.log(finalPlanningOutput.final_answer);
```

**对输出的讨论：**
过程的差异显而易见。第一步是 `Planner` 创建了一个明确且完整的计划，例如：`['web_search("巴黎的人口")', 'web_search("柏林的人口")', ...]`。

随后 Agent 有条不紊地执行了这个计划，而无需在步骤之间停下来思考。这个过程具有以下特点：
- **更透明：** 我们甚至在它开始之前就能看到 Agent 的整体策略。
- **更稳健：** 它不太可能跑偏，因为它正在遵循一套清晰的指令。
- **可能更高效：** 它避免了在步骤之间额外进行 LLM 调用来推理。

这就证明了，对于可以预先确定所需步骤的任务，计划 (Planning) 是非常有威力的。

## 第 4 阶段：定量评估

为了使我们的比较更加规范，我们将使用“LLM 作为评委 (LLM-as-a-Judge)” 对两个 Agent 进行打分，重点关注它们解决问题过程的质量和效率。

```typescript
// 定义评估 Agent 解决问题过程的数据模式
const ProcessEvaluationSchema = z.object({
  task_completion_score: z.number().describe("1-10分，评价 Agent 是否成功完成了任务。"),
  process_efficiency_score: z.number().describe("1-10分，评价 Agent 过程的效率和直接性。分数越高意味着路径越符合逻辑，绕弯路越少。"),
  justification: z.string().describe("对分数的简短证明/理由。")
});

const judgeLlm = llm.withStructuredOutput(ProcessEvaluationSchema);

const evaluateAgentProcess = async (query: string, finalState: any) => {
  let trace = "";
  
  // 如果是 ReAct Agent，轨迹保存在 'messages' 中；如果是 Planning，则在 'intermediate_steps'
  if (finalState.messages) {
    trace = finalState.messages
      .map((m: any) => `${m._getType()}: ${typeof m.content === "string" ? m.content : JSON.stringify(m.content)}`)
      .join("\n");
  } else {
    trace = `计划内容: ${JSON.stringify(finalState.plan || [])}\n中间步骤: ${JSON.stringify(finalState.intermediate_steps || [])}`;
  }
        
  const prompt = `你是一个人工智能 Agent 的专家评委。请以 1-10 分的等级评估该 Agent 解决任务的过程。
重点关注该过程是否合乎逻辑且高效。
    
**用户的任务:** ${query}
**Agent 完整执行轨迹:**
\`\`\`
${trace}
\`\`\`
`;
  return await judgeLlm.invoke(prompt);
};

console.log(chalk.bold("\n--- 评估反应式 Agent (ReAct) 的过程 ---"));
const reactAgentEvaluation = await evaluateAgentProcess(planCentricQuery, finalReactOutput);
console.dir(reactAgentEvaluation, { depth: null });

console.log(chalk.bold("\n--- 评估计划型 Agent (Planning) 的过程 ---"));
const planningAgentEvaluation = await evaluateAgentProcess(planCentricQuery, finalPlanningOutput);
console.dir(planningAgentEvaluation, { depth: null });
```

**对输出的讨论：**
评委的分数量化了两种方法之间的差异。两个 Agent 都有可能获得很高的 `task_completion_score` (任务完成分)，因为它们最终都能找到答案。但是，**计划型 Agent** 将获得明显更高的 `process_efficiency_score` (流程效率分)。评委的理由会强调，与其像 ReAct Agent 那样步步为营的探索性过程相比，前期计划是解决问题更直接、更合乎逻辑的方法。

这项评估证实了我们的假设：对于解决方案路径可以预测的问题，计划 (Planning) 架构提供了一种更结构化、透明且高效的方法。

## 结论

在本文档中，我们实现了**计划 (Planning)** 架构，并将其与 **ReAct** 模式进行了直接对比。通过迫使 Agent 在执行前首先构建全面计划，我们在面对定义清晰、多步骤的任务时，获得了系统透明度、鲁棒性以及效率的显著提升。

当 ReAct 在下一步未知、需要探索场景下表现优异时，如果在进入行动前就可以规划出一条达成目标的通途，计划模式则是当仁不让的最优选。对于系统设计者而言，理解这种权衡至关重要。选择与问题匹配的架构是构建高效且智能的 AI Agent 的核心能力。计划模式是该工具箱里不可或缺的一件利器，它为复杂但可预测的工作流程提供了必要的结构化骨架。