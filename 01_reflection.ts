import chalk from "chalk";


// ─────────────────────────────────────────────────────────────
// 1. 环境变量与追踪设置 (Bun 原生支持 .env，无需 dotenv 包)
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
process.env.LANGCHAIN_PROJECT = "Agentic Architecture - Reflection (Nebius)";
if (LANGCHAIN_API_KEY) process.env.LANGCHAIN_API_KEY = LANGCHAIN_API_KEY;

console.log(chalk.green("✅ 环境变量已加载，追踪设置完毕。"));

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

import { ChatOpenAI } from "@langchain/openai";

// 初始化连接 ModelScope 的 LLM
const llm = new ChatOpenAI({
  modelName: "deepseek-ai/DeepSeek-V4-Flash", // 这里替换为魔搭支持的模型
  temperature: 0.2,
  configuration: {
    baseURL: process.env.OPENAI_API_BASE,
  },
});

import { Annotation, StateGraph, START, END } from "@langchain/langgraph";

// 定义状态图的结构
const ReflectionState = Annotation.Root({
  user_request: Annotation<string>(),
  draft: Annotation<z.infer<typeof draftCodeSchema>>(),
  critique: Annotation<z.infer<typeof critiqueSchema>>(),
  refined_code: Annotation<z.infer<typeof refinedCodeSchema>>(),
});

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

async function main() {
  const userRequest = "编写一个 JS 函数来查找第 n 个斐波那契数列。";
  const initialInput = { user_request: userRequest };

  console.log(chalk.bold.magenta(`\n🚀 启动 Reflection 工作流: '${userRequest}'\n`));
  // 运行并监听图的状态流转
  const finalState = await reflectionApp.invoke(initialInput);

  console.log(chalk.bold.green("\n✅ Reflection 工作流执行完毕!\n"));

  console.log(JSON.stringify(finalState, null, 2));

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

    // --- 步骤 3.3：评估代码 ---
    console.log(chalk.bold.blue("\n🔎 评估初始代码质量..."));
    const initialEvaluation = await evaluateCode(finalState.draft.code);
    console.log(chalk.bold.blue("初始代码评分:"), initialEvaluation);
    console.log(chalk.bold.blue("\n🔎 评估最终代码质量..."));
    const finalEvaluation = await evaluateCode(finalState.refined_code.refined_code);
    console.log(chalk.bold.blue("最终代码评分:"), finalEvaluation);
  }
}

main().catch(console.error);