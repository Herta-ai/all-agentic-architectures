以下是将该 Notebook 翻译为中文，并将 Python 代码转换为 JavaScript (使用 `langchain.js` 和 `langgraph.js`) 的完整内容。

---

# 📘 智能体架构 5：多智能体系统 (Multi-Agent Systems)

在本笔记本中，我们将深入探讨最强大、最灵活的架构之一：**多智能体系统 (Multi-Agent System)**。这种模式超越了单一智能体的概念（无论单一智能体有多复杂），而是构建一个由专业智能体组成的团队，协同解决问题。每个智能体都有独特的角色、人设和技能集，这与人类专家团队的工作方式如出一辙。

这种方法实现了深度的“劳动分工”，将复杂问题分解为子任务，并分配给最适合该任务的智能体。为了展示其强大之处，我们将进行直接对比。首先，我们将让一个**单体“通用型”智能体 (monolithic 'generalist' agent)** 创建一份全面的市场分析报告。然后，我们将组建一个**专家团队 (specialist team)**——包括一名技术分析师、一名新闻分析师和一名财务分析师——并让第四个“经理”智能体将他们的专家意见综合成最终报告。质量、结构和深度的差异将显而易见。

### 定义
**多智能体系统**是一种架构，其中一组不同的、专业的智能体协同（有时是竞争）以实现共同目标。系统通过中央控制器或定义好的工作流协议来管理通信并在智能体之间路由任务。

### 高层工作流

1.  **分解 (Decomposition)：** 主控制器或用户提供一个复杂的任务。
2.  **角色定义 (Role Definition)：** 系统根据定义的角色（例如“研究员”、“程序员”、“评论员”、“作家”）将子任务分配给专业的智能体。
3.  **协作 (Collaboration)：** 智能体执行各自的任务，通常是并行或串行的。它们将输出传递给彼此或传递到一个中央“黑板 (blackboard)”。
4.  **综合 (Synthesis)：** 最终的“经理”或“综合者”智能体收集专家智能体的输出，并组装出最终的综合响应。

### 何时使用 / 应用场景
*   **复杂报告生成：** 创建需要多个领域专业知识的详细报告（例如，财务分析、科学研究）。
*   **软件开发流水线：** 模拟一个包含程序员、代码审查员、测试员和项目经理的开发团队。
*   **创意头脑风暴：** 一个具有不同“性格”的智能体团队（例如，一个乐观、一个谨慎、一个极具创造力）可以产生更多样化的想法。

### 优势与劣势
*   **优势：**
    *   **专业性与深度：** 每个智能体都可以通过特定的人设和工具进行微调，从而在其领域内产生更高质量的工作。
    *   **模块化与可扩展性：** 很容易添加、删除或升级单个智能体，而无需重新设计整个系统。
    *   **并行性：** 多个智能体可以同时处理其子任务，从而有可能减少总体任务时间。
*   **劣势：**
    *   **协调开销：** 管理智能体之间的通信和工作流增加了系统设计的复杂性。
    *   **成本与延迟增加：** 运行多个智能体涉及更多的 LLM 调用，这可能比单智能体方法更昂贵且更慢。

## 阶段 0：基础与设置

我们将从安装依赖库和配置 Nebius、LangSmith 和 Tavily 的 API 密钥开始。

### 步骤 0.1：安装核心库

**我们要做什么：**
我们将安装本系列项目所需的标准库套件。*(注：在 Node.js 环境中，我们使用 npm)*

```bash
npm install @langchain/core @langchain/openai @langchain/community @langchain/langgraph dotenv zod
```

### 步骤 0.2：导入库并设置密钥

**我们要做什么：**
我们将导入必要的模块并从 `.env` 文件加载 API 密钥。

**需要执行的操作：** 在当前目录下创建一个 `.env` 文件，填入你的密钥：
```env
NEBIUS_API_KEY="your_nebius_api_key_here"
LANGCHAIN_API_KEY="your_langsmith_api_key_here"
TAVILY_API_KEY="your_tavily_api_key_here"
```

```javascript
import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StateGraph, END, Annotation, messagesStateReducer } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { z } from "zod";

// --- API 密钥和 Tracing 设置 ---
process.env.LANGCHAIN_TRACING_V2 = "true";
process.env.LANGCHAIN_PROJECT = "Agentic Architecture - Multi-Agent (Nebius JS)";

const requiredKeys = ["NEBIUS_API_KEY", "LANGCHAIN_API_KEY", "TAVILY_API_KEY"];
for (const key of requiredKeys) {
    if (!process.env[key]) {
        console.log(`未找到 ${key}。请创建 .env 文件并进行设置。`);
    }
}

console.log("环境变量已加载，追踪已设置。");
```

**输出：**
```text
环境变量已加载，追踪已设置。
```

## 阶段 1：基准 - 单体“通用型”智能体

为了展示专家团队的价值，我们首先需要看看单一智能体在复杂任务上的表现。我们将构建一个 ReAct 智能体，并给它一个宽泛的提示词，要求它同时执行多种类型的分析。

### 步骤 1.1：构建单体智能体

**我们要做什么：**
我们将构建一个标准的 ReAct 智能体。我们将为它提供一个网络搜索工具和一个非常通用的系统提示词，要求它成为一名全面的财务分析师。

```javascript
// 定义智能体的共享状态
const AgentState = Annotation.Root({
    messages: Annotation({
        reducer: messagesStateReducer,
        default: () => [],
    }),
});

// 定义工具和 LLM
const searchTool = new TavilySearchResults({ maxResults: 3, name: "web_search" });

// 注意：在 JS 中，我们使用配置了 baseURL 的 ChatOpenAI 来调用兼容 OpenAI 格式的 Nebius API
const llm = new ChatOpenAI({
    modelName: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    temperature: 0,
    openAIApiKey: process.env.NEBIUS_API_KEY,
    configuration: {
        baseURL: "https://api.nebius.com/v1",
    }
});
const llmWithTools = llm.bindTools([searchTool]);

// 定义单体智能体节点
async function monolithicAgentNode(state) {
    console.log("--- 单体智能体：思考中... ---");
    const response = await llmWithTools.invoke(state.messages);
    return { messages: [response] };
}

const toolNode = new ToolNode([searchTool]);

// 构建单体智能体的 ReAct 图
const monoGraphBuilder = new StateGraph(AgentState)
    .addNode("agent", monolithicAgentNode)
    .addNode("tools", toolNode)
    .addEdge("__start__", "agent")
    .addConditionalEdges("agent", toolsCondition)
    .addEdge("tools", "agent");

const monolithicAgentApp = monoGraphBuilder.compile();

console.log("单体'通用型'智能体编译成功。");
```

**输出：**
```text
单体'通用型'智能体编译成功。
```

### 步骤 1.2：测试单体智能体

**我们要做什么：**
我们将给通用型智能体一个复杂的任务：为一家公司创建一份完整的市场分析报告，涵盖三个不同的领域。

```javascript
const company = "NVIDIA (NVDA)";
const monolithicQuery = `为 ${company} 创建一份简短但全面的市场分析报告。报告应包括三个部分：1. 近期新闻和市场情绪总结。2. 股票价格趋势的基本技术分析。3. 公司近期财务表现概览。`;

console.log(`\n测试单体智能体执行多方面任务:\n'${monolithicQuery}'\n`);

const finalMonoOutput = await monolithicAgentApp.invoke({
    messages: [
        new SystemMessage("你是一个全能的专家级金融分析师。你必须创建一份涵盖用户请求所有方面的综合报告。"),
        new HumanMessage(monolithicQuery)
    ]
});

console.log("\n--- 单体智能体的最终报告 ---");
console.log(finalMonoOutput.messages[finalMonoOutput.messages.length - 1].content);
```

**输出讨论：**
单体智能体生成了一份报告。它可能执行了几次网络搜索，并尽力综合了这些信息。然而，输出可能存在一些弱点：
- **缺乏结构：** 各个部分可能混杂在一起，没有清晰的标题或专业的格式。
- **分析肤浅：** 试图同时成为三个领域的专家，智能体可能只提供高层次的总结，而在任何单一领域都没有足够的深度。
- **语气平淡：** 语言可能比较通用，缺乏每个领域真正专家特有的专业术语和焦点。

这个结果是我们的基准。它能起作用，但并不出色。接下来，我们将组建一个专家团队，看看是否能做得更好。

## 阶段 2：进阶方法 - 多智能体专家团队

现在我们将组建我们的团队：一名新闻分析师、一名技术分析师和一名财务分析师。每个分析师都将是具有特定人设的独立智能体节点。最后，一名报告撰写员将担任经理，汇总他们的工作。

### 步骤 2.1：定义专家智能体节点

**我们要做什么：**
我们将创建三个不同的智能体节点。关键的区别在于我们给每个节点的系统提示词非常具体。这个提示词定义了他们的人设、专业领域以及输出应采用的精确格式。这就是我们强制实现专业化的方式。

```javascript
// 多智能体系统的状态将保存每个专家的输出
const MultiAgentState = Annotation.Root({
    user_request: Annotation(),
    news_report: Annotation(),
    technical_report: Annotation(),
    financial_report: Annotation(),
    final_report: Annotation(),
});

function createSpecialistNode(persona, outputKey) {
    /** 创建专家智能体节点的工厂函数 */
    const systemPrompt = `${persona}\n\n你可以使用网络搜索工具。你的输出必须是一个简洁的报告部分，使用 markdown 格式，并且只关注你的专业领域。`;

    const promptTemplate = ChatPromptTemplate.fromMessages([
        ["system", systemPrompt],
        ["human", "{user_request}"]
    ]);

    const agent = promptTemplate.pipe(llmWithTools);

    return async function specialistNode(state) {
        console.log(`--- 调用 ${outputKey.replace('_report', '').toUpperCase()} 分析师 ---`);
        const result = await agent.invoke({ user_request: state.user_request });
        
        // 如果没有直接内容（例如模型决定调用工具），则记录工具调用
        let content = result.content;
        if (!content && result.tool_calls?.length > 0) {
            content = `无直接文本内容，工具调用记录: ${JSON.stringify(result.tool_calls)}`;
        }
        
        return { [outputKey]: content };
    };
}

// 创建专家节点
const newsAnalystNode = createSpecialistNode(
    "你是一位资深的新闻分析师。你的专长是在网络上搜索有关公司的最新新闻、文章和社交媒体情绪。",
    "news_report"
);
const technicalAnalystNode = createSpecialistNode(
    "你是一位资深的技术分析师。你擅长分析股票价格图表、趋势和技术指标。",
    "technical_report"
);
const financialAnalystNode = createSpecialistNode(
    "你是一位资深的财务分析师。你擅长解读财务报表和绩效指标。",
    "financial_report"
);

async function reportWriterNode(state) {
    /** 综合专家报告的经理智能体 */
    console.log("--- 调用报告撰写员 ---");
    const prompt = `你是一位资深的金融编辑。你的任务是将以下专家报告整合成一份单一的、专业的、连贯的市场分析报告。请添加简短的引言和结论段落。
    
    新闻与情绪报告:
    ${state.news_report}
    
    技术分析报告:
    ${state.technical_report}
    
    财务表现报告:
    ${state.financial_report}
    `;
    const result = await llm.invoke(prompt);
    return { final_report: result.content };
}

console.log("专家智能体节点和报告撰写员节点已定义。");
```

**输出：**
```text
专家智能体节点和报告撰写员节点已定义。
```

### 步骤 2.2：构建多智能体图

**我们要做什么：**
现在我们将专家和经理连接成一个图。对于这个任务，专家可以独立工作，因此我们可以按简单的顺序运行它们（在实际应用中，这些可以并行运行）。最后一步始终是报告撰写员。

```javascript
const multiAgentGraphBuilder = new StateGraph(MultiAgentState)
    // 添加所有节点
    .addNode("news_analyst", newsAnalystNode)
    .addNode("technical_analyst", technicalAnalystNode)
    .addNode("financial_analyst", financialAnalystNode)
    .addNode("report_writer", reportWriterNode)
    // 定义工作流顺序
    .addEdge("__start__", "news_analyst")
    .addEdge("news_analyst", "technical_analyst")
    .addEdge("technical_analyst", "financial_analyst")
    .addEdge("financial_analyst", "report_writer")
    .addEdge("report_writer", END);

const multiAgentApp = multiAgentGraphBuilder.compile();
console.log("多智能体专家团队编译成功。");
```

**输出：**
```text
多智能体专家团队编译成功。
```

## 阶段 3：正面交锋对比

现在，我们将让专家团队执行与单体智能体完全相同的任务，并比较最终报告。

```javascript
const multiAgentQuery = `为 ${company} 创建一份简短但全面的市场分析报告。`;
const initialMultiAgentInput = { user_request: multiAgentQuery };

console.log(`\n测试多智能体团队执行相同任务:\n'${multiAgentQuery}'\n`);

const finalMultiAgentOutput = await multiAgentApp.invoke(initialMultiAgentInput);

console.log("\n--- 多智能体团队的最终报告 ---");
console.log(finalMultiAgentOutput.final_report);
```

**输出讨论：**
最终报告的差异是显著的。多智能体团队的输出特点：
- **高度结构化：** 针对每个分析领域都有清晰、独立的部分，因为每个部分都是由具有特定格式要求的专家生成的。
- **更深入的分析：** 每个部分都包含更详细的、特定领域的语言和见解。技术分析师谈论移动平均线，新闻分析师讨论情绪，财务分析师关注收入和收益。
- **更专业：** 由报告撰写员组装的最终报告读起来像一份专业文件，有清晰的引言、正文和结论。

这种定性比较表明，通过将工作分配给专家团队，我们取得了单体通用型智能体难以复制的卓越成果。

## 阶段 4：定量评估

为了使比较正式化，我们将使用 LLM 作为评委 (LLM-as-a-Judge) 对两份报告进行评分。评分标准将侧重于我们期望在多智能体方法中表现更好的品质，例如结构和分析深度。

```javascript
// 使用 Zod 定义评估的 Schema
const reportEvaluationSchema = z.object({
    clarity_and_structure_score: z.number().int().min(1).max(10)
        .describe("对报告的组织、结构和清晰度进行 1-10 分的打分。"),
    analytical_depth_score: z.number().int().min(1).max(10)
        .describe("对每个部分的分析深度和质量进行 1-10 分的打分。"),
    completeness_score: z.number().int().min(1).max(10)
        .describe("对报告在多大程度上满足了用户请求的所有部分进行 1-10 分的打分。"),
    justification: z.string()
        .describe("对分数的简短理由说明。")
});

// 使用 structured output 绑定 Schema
const judgeLlm = llm.withStructuredOutput(reportEvaluationSchema);

async function evaluateReport(query, report) {
    const prompt = `你是一位评估金融分析报告的专家评委。请根据以下报告的结构、深度和完整性，在 1-10 分的范围内进行打分。
    
    **原始用户请求:**
    ${query}
    
    **要评估的报告:**\n
    ${report}
    `;
    return await judgeLlm.invoke(prompt);
}

console.log("--- 评估单体智能体的报告 ---");
const monoAgentEvaluation = await evaluateReport(
    monolithicQuery, 
    finalMonoOutput.messages[finalMonoOutput.messages.length - 1].content
);
console.log(monoAgentEvaluation);

console.log("\n--- 评估多智能体团队的报告 ---");
const multiAgentEvaluation = await evaluateReport(
    multiAgentQuery, 
    finalMultiAgentOutput.final_report
);
console.log(multiAgentEvaluation);
```

**输出讨论：**
评委的分数提供了我们假设的定量证明。**多智能体团队 (Multi-Agent Team)** 的报告将获得明显更高的分数，特别是在 `clarity_and_structure_score`（清晰度和结构得分）和 `analytical_depth_score`（分析深度得分）方面。评委的理由可能会赞扬清晰的章节划分和每个部分内部详细的专家级分析，这与单体智能体生成的更通用、更混乱的输出形成了鲜明对比。

这项评估证实，对于可以分解为不同专业领域的复杂任务，多智能体架构是生成高质量、结构化和可靠结果的卓越方法。

## 结论

在本笔记本中，我们展示了在处理复杂的多方面任务时，**多智能体系统**相对于单一单体智能体的明显优势。通过创建一个由专业智能体组成的团队，每个智能体都有明确的人设和角色，并由一名经理综合他们的工作，我们生成了质量明显更高的最终输出。

关键的启示是**专业化 (specialization)** 的力量。就像在人类组织中一样，将一个大问题分解并将其各个部分分配给专家会产生更好的结果。虽然这种架构在编排上引入了更多的复杂性，但最终输出在结构、深度和专业性方面的显著提升，使其成为任何需要跨多个领域提供专家级表现的严肃智能体应用不可或缺的模式。