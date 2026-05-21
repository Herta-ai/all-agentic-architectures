# 📘 智能体架构 3：ReAct（推理 + 行动）

欢迎来到我们系列的第三篇笔记。现在我们将探索 **ReAct**，这是一个关键的架构，它弥合了简单的工具调用与复杂的多步问题解决之间的差距。ReAct 代表 **Reason + Act（推理 + 行动）**，其核心创新在于它使智能体能够动态地对问题进行推理，根据推理采取行动，观察结果，然后再进行推理。

这种模式将智能体从静态的“工具调用者”转变为适应性强的“问题解决者”。为了突出它的强大之处，我们将首先构建一个**基础的、单次工具调用的智能体**，并展示它在处理复杂任务时的局限性。然后，我们将构建一个完整的 ReAct 智能体，并演示其迭代的 `思考 -> 行动 -> 观察` 循环如何让它在基础智能体失败的地方取得成功。

### 定义
**ReAct** 架构是一种设计模式，智能体在其中交替进行推理步骤和行动。智能体不会预先计划好所有的步骤，而是生成关于其下一步的“想法（Thought）”，执行一个“行动（Action）”（比如调用工具），“观察（Observe）”结果，然后利用这些新信息生成下一个想法和行动。这创造了一个动态和自适应的循环。

### 高层工作流程

1.  **接收目标：** 智能体被赋予一个复杂的任务。
2.  **思考（推理）：** 智能体生成一个内部想法，例如：*“为了回答这个问题，我首先需要找到信息 X。”*
3.  **行动：** 基于其想法，智能体执行一个操作，通常是调用一个工具（例如，`search_api('X')`）。
4.  **观察：** 智能体接收来自工具的结果。
5.  **重复：** 智能体将观察结果整合到其上下文中，并返回到步骤 2，生成一个新的想法（例如，*“好的，现在我已经有了 X，我需要用它来寻找 Y。”*）。这个循环一直持续到总体目标被满足为止。

### 何时使用 / 应用场景
*   **多跳问答（Multi-hop Question Answering）：** 当回答一个问题需要按顺序查找多条信息时（例如，“制造 iPhone 的公司的现任 CEO 是谁？”）。
*   **网络导航与研究：** 智能体可以搜索一个起点，阅读结果，然后根据学到的内容决定新的搜索查询。
*   **交互式工作流：** 任何环境是动态的、且无法预先知道完整解决方案路径的任务。

### 优势与劣势
*   **优势：**
    *   **自适应与动态：** 可以根据新信息随时调整计划。
    *   **处理复杂性：** 擅长解决需要链接多个依赖步骤的问题。
*   **劣势：**
    *   **较高的延迟与成本：** 涉及多次连续的 LLM 调用，使其比单次调用的方法更慢且更昂贵。
    *   **陷入死循环的风险：** 缺乏良好引导的智能体可能会陷入重复、无效的“思考和行动”循环中。

---

## 阶段 0：基础与设置

我们将从标准的设置过程开始：安装依赖库并配置 API 密钥（LangSmith 和我们的 Tavily 网络搜索工具）。

### 步骤 0.1：安装核心库

**我们要做什么：**
我们将使用 Bun 安装本项目系列所需的标准库套件。

```bash
bun add @langchain/openai @langchain/tavily @langchain/langgraph zod
```

### 步骤 0.2：导入库并设置密钥

**我们要做什么：**
我们将导入必要的模块。Bun 会自动加载当前目录下的 `.env` 文件。

**需要采取的行动：** 在此目录下创建一个 `.env` 文件并填入你的密钥：
```env
OPENAI_BASE_URL=https://api-inference.modelscope.cn/v1
OPENAI_API_KEY="YOUR_NEBIUS_API"
LANGSMITH_TRACING=true
LANGSMITH_API_KEY="your_langsmith_api_key_here" 
TAVILY_API_KEY="YOUR_TAVILY_API"
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { TavilySearch } from "@langchain/tavily";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StateGraph, END, MessagesAnnotation } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { z } from "zod";

// --- API 密钥和 Tracing 设置 ---
// Bun 会自动加载 .env 文件，但我们需要设置 LangSmith 的环境变量
process.env.LANGCHAIN_PROJECT = "Agentic Architecture - ReAct";

const requiredKeys = ["OPENAI_BASE_URL", "OPENAI_API_KEY", "LANGCHAIN_API_KEY", "TAVILY_API_KEY"];
for (const key of requiredKeys) {
  if (!process.env[key]) {
    console.warn(`未找到 ${key}。请在 .env 文件中进行设置。`);
  }
}

console.log("环境变量已加载，Tracing 设置完毕。");
```

## 阶段 1：基础方法 - 单次工具调用者

为了理解为什么 ReAct 如此强大，我们必须首先看看如果没有它会发生什么。我们将构建一个“基础”智能体，它可以调用工具，但**只能调用一次**。它将分析用户的查询，进行一次工具调用，然后尝试基于那一条信息制定最终答案。

### 步骤 1.1：构建基础智能体

**我们要做什么：**
我们将定义工具和 LLM，并将它们连接成一个简单的线性图（Graph）。智能体只有一次调用工具的机会，然后工作流就会结束。这里没有循环。

```typescript
// 定义工具和 LLM
const searchTool = new TavilySearchResults({ maxResults: 2, name: "web_search" });

const llm = new ChatOpenAI({
  modelName: "deepseek-ai/DeepSeek-V4-Flash", // 这里替换为魔搭支持的模型
  temperature: 0.2,
});

const llmWithTools = llm.bindTools([searchTool]);

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
```

### 步骤 1.2：在多步问题上测试基础智能体

**我们要做什么：**
现在我们将给基础智能体一个需要多个依赖步骤才能解决的问题。这将暴露其根本的弱点。

```typescript
const multiStepQuery = "制作科幻电影《沙丘》的公司的现任CEO是谁？该公司最新一部电影的预算是多少？";

console.log(`\n[测试基础智能体处理多步查询]：'${multiStepQuery}'\n`);

const basicAgentOutput = await basicToolAgentApp.invoke({
  messages: [new HumanMessage(multiStepQuery)]
});

console.log("\n--- 基础智能体的最终输出 ---");
const finalBasicMessage = basicAgentOutput.messages[basicAgentOutput.messages.length - 1];
console.log(finalBasicMessage.content);
```

**输出讨论：**
不出所料，基础智能体失败了。它的单次工具调用很可能是直接搜索了整个长句子。对于这种复杂的联合查询，搜索结果通常很杂乱，并且不会在一个地方包含所有必要的信息片段。

智能体的最终答案可能是不完整的、不正确的，或者声明它找不到信息。它无法将问题分解为：
1. 找到制作《沙丘》的公司（传奇影业 / Legendary Entertainment）。
2. 找到该公司的 CEO（Joshua Grode）。
3. 找到该公司最新的一部电影及其预算。

这种失败完美地说明了我们需要一种更动态的方法。智能体需要一种方法来对它在一步中发现的信息做出**反应（react）**，从而指导下一步。

---

## 阶段 2：高级方法 - 实现 ReAct

现在，我们将构建真正的 ReAct 智能体。核心区别在于图（Graph）的结构：我们将引入一个循环，允许智能体重复地思考、行动和观察。

### 步骤 2.1：构建 ReAct 智能体图

**我们要做什么：**
我们将定义节点和至关重要的路由函数，以创建 `思考 -> 行动` 循环。关键的架构变化是从 `tools` 节点路由回 `agent` 节点的边，这使得智能体能够看到结果并决定其下一步。

```typescript
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
```

---

## 阶段 3：正面交锋对比

现在，我们将使用我们新的 ReAct 智能体运行相同的复杂查询，并观察其过程和最终输出的差异。

### 步骤 3.1：在多步问题上测试 ReAct 智能体

**我们要做什么：**
我们将使用相同的多步查询调用 ReAct 智能体，并以流（Stream）的方式输出，以查看其迭代推理过程。

```typescript
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
  console.log(`[${lastMsg._getType()}]`, lastMsg.content || "(Tool Call / Empty Content)");
  if (lastMsg.tool_calls && lastMsg.tool_calls.length > 0) {
    console.log("工具调用:", JSON.stringify(lastMsg.tool_calls, null, 2));
  }
  console.log("\n");
}

console.log("\n--- ReAct 智能体的最终输出 ---");
const finalReactMessage = finalReactOutput.messages[finalReactOutput.messages.length - 1];
console.log(finalReactMessage.content);
```

**输出讨论：**
成功！执行追踪显示了一个完全不同且更加智能的过程。你可以看到智能体循序渐进的推理：
1.  **想法 1：** 它首先推断需要确定《沙丘》的制作公司。
2.  **行动 1：** 它调用 `web_search` 工具，查询类似“《沙丘》电影的制作公司”。
3.  **观察 1：** 它收到结果：“传奇影业（Legendary Entertainment）”。
4.  **想法 2：** 现在，结合新信息，它推断需要寻找传奇影业的 CEO。
5.  **行动 2：** 它再次调用 `web_search`，查询类似“传奇影业 CEO”。
6.  ...以此类推，直到它收集到所有必要的信息片段。
7.  **综合：** 最后，它将所有收集到的事实组装成一个完整且准确的答案。

这清楚地证明了对于任何非简单的、需要多步查找的任务，ReAct 模式都具有优越性。

---

## 阶段 4：定量评估

为了使比较正式化，我们将使用“LLM 作为裁判（LLM-as-a-Judge）”来对基础智能体和 ReAct 智能体的最终输出进行评分，评估它们完成任务的能力。

```typescript
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

console.log("--- 评估基础智能体的输出 ---");
const basicAgentEvaluation = await evaluateAgentOutput(multiStepQuery, basicAgentOutput);
console.log(JSON.stringify(basicAgentEvaluation, null, 2));

console.log("\n--- 评估 ReAct 智能体的输出 ---");
const reactAgentEvaluation = await evaluateAgentOutput(multiStepQuery, finalReactOutput);
console.log(JSON.stringify(reactAgentEvaluation, null, 2));
```

**输出讨论：**
来自“LLM 裁判”的定量评分让差异变得非常清晰。
- **基础智能体**的 `task_completion_score`（任务完成得分）非常低，因为它未能收集到所有必需的信息。它的 `reasoning_quality_score`（推理质量得分）也很低，因为它的过程存在缺陷且不完整。
- 相比之下，**ReAct 智能体**获得了近乎完美的满分。裁判认识到其迭代过程使其能够成功完成复杂任务的所有部分。

这种正面交锋的比较和评估为 ReAct 架构的价值提供了决定性的证据。它是解锁智能体解决需要动态适应的复杂、多跳问题能力的关键。

---

## 结论

在这篇笔记中，我们不仅实现了 **ReAct** 架构，还展示了它相对于更基础的单次调用方法的明显优势。通过构建一个允许智能体在推理和行动的循环中迭代的工作流，我们使其能够解决原本难以处理的复杂、多步问题。

观察行动结果并利用该信息指导下一步的能力是智能行为的基本组成部分。ReAct 模式提供了一种简单而极其有效的方法，将这种能力构建到我们的 AI 智能体中，使它们变得更强大、更具适应性，并在现实世界的任务中更加有用。