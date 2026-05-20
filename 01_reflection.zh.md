# 📘 智能体架构 1：反思 (Reflection) - TypeScript 版本

欢迎来到我们深入探讨关键智能体架构的第一本笔记。我们从最基础、最强大的模式之一开始：**反思 (Reflection)**。

这种模式将大语言模型 (LLM) 从简单的单次生成器提升为更加审慎和稳健的推理器。反思型智能体不会仅仅提供它想到的第一个答案，而是会退一步来批判、分析和完善自己的工作。这种自我改进的迭代过程是构建更可靠、更高质量 AI 系统的基石。

### 高级工作流

1. **生成 (Generate)：** 智能体根据用户的提示生成初始草稿或解决方案。
2. **批判 (Critique)：** 智能体随后转换角色成为批评者，寻找逻辑缺陷、Bug或改进空间。
3. **完善 (Refine)：** 利用自我批判中得出的洞察，智能体生成最终的、改进后的输出版本。

---

## 阶段 0：基础与环境设置

在构建我们的反思型智能体之前，我们需要设置我们的 Bun 环境。这包括安装必要的包以及配置我们的 API 密钥。

### 步骤 0.1：安装核心依赖

**我们将要做什么：**
我们将安装该项目必需的包。`@langchain/openai` 允许我们连接兼容 OpenAI 格式的 API（例如魔搭 ModelScope），`@langchain/langgraph` 提供核心编排，`zod` 用于定义结构化数据（相当于 Python 的 Pydantic），`chalk` 用于美化终端输出。

```bash
bun add @langchain/core @langchain/openai @langchain/langgraph zod chalk
```

### 步骤 0.2：导入库和设置密钥

**需要执行的操作：** 在项目根目录创建一个名为 `.env` 的文件，并将您的魔搭 ModelScope 密钥添加到其中：

```env
OPENAI_API_BASE="https://api-inference.modelscope.cn/v1"
OPENAI_API_KEY="your_modelscope_api_key_here"
LANGCHAIN_API_KEY="your_langsmith_api_key_here"
```

新建 `index.ts` 并写入以下基础设置代码：

```typescript
import chalk from "chalk";

// ─────────────────────────────────────────────────────────────
// 1. 环境变量与追踪设置 (Bun 原生支持读取 .env)
// ─────────────────────────────────────────────────────────────
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const LANGCHAIN_API_KEY = process.env.LANGCHAIN_API_KEY;

if (!OPENAI_API_KEY) {
  console.error(chalk.red("❌ 未找到 OPENAI_API_KEY。请创建 .env 文件并设置。"));
  process.exit(1);
}
if (!LANGCHAIN_API_KEY) {
  console.warn(chalk.yellow("⚠️ 未找到 LANGCHAIN_API_KEY。LangSmith 追踪将不可用。"));
}

// LangSmith 追踪配置
process.env.LANGCHAIN_TRACING_V2 = "true";
process.env.LANGCHAIN_PROJECT = "Agentic Architecture - Reflection (TS)";

console.log(chalk.green("✅ 环境变量已加载，准备就绪。"));
```

---

## 阶段 1：构建反思机制的核心组件

我们将把它构建为一个结构化的、由三部分组成的系统：**生成器 (Generator)**、**批评者 (Critic)** 和 **完善者 (Refiner)**。为了确保可靠性，我们将使用 **Zod** 来定义每个步骤的预期输出模式 (Schema)。

### 步骤 1.1：使用 Zod 定义数据模式

我们将定义 Zod 模式，作为我们 LLM 的输出契约。

```typescript
import { z } from "zod";

const draftCodeSchema = z.object({
  code: z.string().describe("为解决用户请求而生成的 TypeScript/JavaScript/Python 等代码。"),
  explanation: z.string().describe("对代码如何运行的简短解释。"),
});

const critiqueSchema = z.object({
  has_errors: z.boolean().describe("代码是否有任何潜在的 Bug 或逻辑错误？"),
  is_efficient: z.boolean().describe("代码的编写方式是否高效且最优？"),
  suggested_improvements: z.array(z.string()).describe("具体、可操作的代码改进建议。"),
  critique_summary: z.string().describe("批判的总结。"),
});

const refinedCodeSchema = z.object({
  refined_code: z.string().describe("最终的、改进后的代码。"),
  refinement_summary: z.string().describe("基于批判所做更改的总结。"),
});

console.log("✅ Zod 数据模式定义完成。");
```

### 步骤 1.2：初始化模型

使用 `@langchain/openai` 连接 ModelScope 的免费模型（一天一个模型500次+一共2000次调用）

```typescript
import { ChatOpenAI } from "@langchain/openai";

// 初始化连接 ModelScope 的 LLM
const llm = new ChatOpenAI({
  modelName: "deepseek-ai/DeepSeek-V4-Flash", // 这里替换为魔搭支持的模型
  temperature: 0.2,
  configuration: {
    baseURL: process.env.OPENAI_API_BASE,
  },
});
```

### 步骤 1.3：创建节点 (Nodes)

接下来，我们编写对应生成、批判和完善的三个函数。在 LangGraph JS 中，节点函数接收当前的状态对象，并返回需要合并到状态中的新对象。

```typescript
// --- 1. 生成器节点 ---
async function generatorNode(state: typeof ReflectionState.State) {
  console.log(chalk.cyan("--- 1. 生成初始草稿 ---"));
  
  const generatorLlm = llm.withStructuredOutput(draftCodeSchema, { name: "DraftCode" });
const prompt = `你是一位资深工程师。请针对以下需求编写实现函数。
代码需保持简洁清晰，并附上核心逻辑说明。

需求：${state.user_request}`;

  const draft = await generatorLlm.invoke(prompt);
  return { draft };
}

// --- 2. 批评者节点 ---
async function criticNode(state: typeof ReflectionState.State) {
  console.log(chalk.cyan("--- 2. 批判草稿 ---"));
  
  const criticLlm = llm.withStructuredOutput(critiqueSchema, { name: "Critique" });
  const codeToCritique = state.draft?.code;

const prompt = `你是一位资深代码审查专家。请对以下代码进行深度审查。
重点分析：
1. 缺陷与错误：是否存在边界情况遗漏或逻辑漏洞？
2. 性能效率：当前实现是否为最优解（时间/空间复杂度）？

待审查代码：
\`\`\`
${codeToCritique}
\`\`\``;

  const critique = await criticLlm.invoke(prompt);
  return { critique };
}

// --- 3. 完善者节点 ---
async function refinerNode(state: typeof ReflectionState.State) {
  console.log(chalk.cyan("--- 3. 完善代码 ---"));
  
  const refinerLlm = llm.withStructuredOutput(refinedCodeSchema, { name: "RefinedCode" });
  const draftCode = state.draft?.code;
  const critiqueSuggestions = JSON.stringify(state.critique, null, 2);

const prompt = `你是一位资深工程师，负责根据审查意见重构代码。
请重写原始代码，全面落实审查中提出的所有改进建议。

**原始代码：**
\`\`\`
${draftCode}
\`\`\`

**审查意见与改进建议：**
${critiqueSuggestions}

请输出最终优化后的代码，并附上变更摘要。`;

  const refined_code = await refinerLlm.invoke(prompt);
  return { refined_code };
}
```

---

## 阶段 2：使用 LangGraph JS 编排工作流

### 步骤 2.1：定义图状态 (Graph State)

我们将使用 LangGraph 的 `Annotation` 机制定义图的数据流结构（等同于 Python 的 `TypedDict`）。

```typescript
import { Annotation, StateGraph, START, END } from "@langchain/langgraph";

// 定义状态图的结构
const ReflectionState = Annotation.Root({
  user_request: Annotation<string>(),
  draft: Annotation<z.infer<typeof draftCodeSchema>>(),
  critique: Annotation<z.infer<typeof critiqueSchema>>(),
  refined_code: Annotation<z.infer<typeof refinedCodeSchema>>(),
});
```

### 步骤 2.2：构建和编译图

我们将刚才创建的三个节点连接成一个简单的线性流水线。

```typescript
const graphBuilder = new StateGraph(ReflectionState)
  .addNode("generator", generatorNode)
  .addNode("critic", criticNode)
  .addNode("refiner", refinerNode)
  .addEdge(START, "generator")
  .addEdge("generator", "critic")
  .addEdge("critic", "refiner")
  .addEdge("refiner", END);

const reflectionApp = graphBuilder.compile();
console.log(chalk.green("✅ Reflection graph 编译成功！"));
```

---

## 阶段 3：端到端执行与评估

### 步骤 3.1：运行工作流

我们将要求模型写一个计算斐波那契数列的函数。这是一个经典的例子，因为模型通常初稿会写出性能极差的递归算法，然后在自我反思阶段优化为迭代或动态规划算法。

```typescript
async function main() {
  const userRequest = "编写一个 JS 函数来查找第 n 个斐波那契数列。";
  const initialInput = { user_request: userRequest };

  console.log(chalk.bold.magenta(`\n🚀 启动 Reflection 工作流: '${userRequest}'\n`));

  let finalState: typeof ReflectionState.State | undefined;

  // 运行并监听图的状态流转
  const stream = await reflectionApp.stream(initialInput);
  for await (const stateUpdate of stream) {
    // 捕获最后一次更新的状态合并结果
    finalState = { ...finalState, ...stateUpdate };
  }

  console.log(chalk.bold.green("\n✅ Reflection 工作流执行完毕!\n"));

  // --- 步骤 3.2：打印对比结果 ---
  if (finalState && finalState.draft && finalState.critique && finalState.refined_code) {
    console.log(chalk.bgBlue.white.bold("\n --- 📝 初始草稿 (Initial Draft) --- "));
    console.log(chalk.bold("解释:"), finalState.draft.explanation);
    console.log(chalk.gray(finalState.draft.code));

    console.log(chalk.bgRed.white.bold("\n --- 🔍 批判意见 (Critique) --- "));
    console.log(chalk.bold("总结:"), finalState.critique.critique_summary);
    console.log(chalk.bold("存在错误:"), finalState.critique.has_errors, "|", chalk.bold("是否高效:"), finalState.critique.is_efficient);
    console.log(chalk.bold("改进建议:"));
    finalState.critique.suggested_improvements.forEach((imp: string) => console.log(`- ${imp}`));

    console.log(chalk.bgGreen.white.bold("\n --- ✨ 最终代码 (Final Refined Code) --- "));
    console.log(chalk.bold("修改总结:"), finalState.refined_code.refinement_summary);
    console.log(chalk.greenBright(finalState.refined_code.refined_code));
  }
}

main().catch(console.error);
```

### 步骤 3.3：定量评估 (LLM-as-a-Judge)

*（您可以将此函数添加到 `main` 中以定量测试代码前后的改进程度）*

```typescript
const evaluationSchema = z.object({
  correctness_score: z.number().describe("逻辑是否正确的 1-10 分评分。"),
  efficiency_score: z.number().describe("算法效率的 1-10 分评分。"),
  style_score: z.number().describe("代码风格和可读性的 1-10 分评分。"),
  justification: z.string().describe("评分的简短理由。"),
});

async function evaluateCode(codeToEvaluate: string) {
  const judgeLlm = llm.withStructuredOutput(evaluationSchema, { name: "CodeEvaluation" });
  const prompt = `你是一位资深代码评审专家。请从正确性、执行效率与代码风格三个维度，对以下函数进行 1-10 分制评分，并附上简要的评判依据。
  
代码:
\`\`\`
${codeToEvaluate}
\`\`\``;

  return await judgeLlm.invoke(prompt);
}
```

加到`main`最后：

```
console.log(chalk.bold.blue("\n🔎 评估初始代码质量..."));
const initialEvaluation = await evaluateCode(finalState.draft.code);
console.log(chalk.bold.blue("初始代码评分:"), initialEvaluation);
console.log(chalk.bold.blue("\n🔎 评估最终代码质量..."));
const finalEvaluation = await evaluateCode(finalState.refined_code.refined_code);
console.log(chalk.bold.blue("最终代码评分:"), finalEvaluation);
```

预期输出：

```
🔎 评估初始代码质量...
初始代码评分: {
  correctness_score: 10,
  efficiency_score: 9,
  style_score: 10,
  justification: "正确性：满分。代码正确处理了边界情况（n<0返回undefined，n=0返回0，n=1返回1），迭代逻辑正确，能够准确计算斐波那契数。\n效率：9分。采用迭代方式，时间复杂度O(n)，空间复杂度O(1)，避免递归栈溢出，非常高效。扣1分是因为JavaScript中Number类型对于大数（如n>79）会出现精度丢失或溢出（超过Number.MAX_SAFE_INTEGER），但题目未要求处理大数，且常见场景下性能已最优。\n代码风格：10分。代码简洁清晰，变量命名有意义，注释恰当，格式规范，符合良好的编码习惯。",
}

🔎 评估最终代码质量...
最终代码评分: {
  correctness_score: 10,
  efficiency_score: 8,
  style_score: 9,
  justification: "正确性: 完美的边界处理与类型一致性，BigInt 保证任意 n 的精确计算，无纰漏。执行效率: 标准 O(n) 迭代，虽无法使用矩阵快速幂等优化，但对常规使用场景已足够，可接受扣1分因未考虑极大规模优化。代码风格: 清晰注释、统一返回类型、输入验证完善，风格优秀扣1分因缺少局部变量 const 关键字（let next 可改为 const）及可选的 Number 转换注释略显冗余。",
}
```

---

## 运行方式

由于我们使用的是 Bun，只需在终端中执行：

```bash
bun run index.ts
```

### 总结

通过这段 TypeScript 代码，您利用 `LangGraph JS` 结构化了 AI 的思考过程。与单次请求不同，Zod 确保了状态（草稿 -> 批判 -> 最终版）能够完美接力。您会看到它通常会抓出自己初始代码中 $O(2^n)$ 指数级时间复杂度的低效递归（斐波那契数），并在最终阶段完美重写为 $O(n)$ 的线性算法。这就是**反思设计模式**的核心威力！