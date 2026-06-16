import { ChatOpenAI } from '@langchain/openai'
import { TavilySearch } from '@langchain/tavily'
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages'
import {
  StateGraph,
  START,
  END,
  MessagesAnnotation,
  Annotation,
} from '@langchain/langgraph'
import { ToolNode, toolsCondition } from '@langchain/langgraph/prebuilt'
import { z } from 'zod'
import chalk from 'chalk'

// --- API 密钥和 Tracing 设置 ---
// Bun 会自动加载 .env 文件，但我们需要设置 LangSmith 的环境变量
process.env.LANGCHAIN_PROJECT = 'Agentic Architecture - ReAct'

const requiredKeys = [
  'OPENAI_BASE_URL',
  'OPENAI_API_KEY',
  'LANGSMITH_TRACING',
  'TAVILY_API_KEY',
]
for (const key of requiredKeys) {
  if (!process.env[key]) {
    console.warn(`未找到 ${key}。请在 .env 文件中进行设置。`)
  }
}

console.log('环境变量已加载，Tracing 设置完毕。')

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

// 1. 使用 Zod 定义确保计划员输出是结构化步骤列表的模式
const PlanSchema = z.object({
  steps: z
    .array(z.string())
    .describe('一系列工具调用的列表。一旦全部执行完毕，将能回答查询。'),
})

// 2. 为计划型 Agent 定义图状态 (State)
const PlanningStateAnnotation = Annotation.Root({
  user_request: Annotation<string>,
  original_plan: Annotation<string[]>, // 保留完整计划供事后评估
  plan: Annotation<string[]>({
    reducer: (state, update) => update, // 覆盖更新计划数组
  }),
  intermediate_steps: Annotation<ToolMessage[]>({
    reducer: (state, update) => state.concat(update), // 累加中间步骤
    default: () => [],
  }),
  final_answer: Annotation<string>,
})

// 3. 计划员节点
const plannerNode = async (state: typeof PlanningStateAnnotation.State) => {
  console.log(chalk.cyan('--- 计划员 (PLANNER): 正在分解任务... ---'))
  const plannerLlm = llm.withStructuredOutput(PlanSchema, {
    method: 'jsonMode',
  })

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
${state.user_request}`

  const planResult = await plannerLlm.invoke(prompt)
  console.log(
    chalk.cyan(
      `--- 计划员 (PLANNER): 生成的计划: [${planResult.steps.join(', ')}] ---`,
    ),
  )

  return { plan: planResult.steps, original_plan: planResult.steps }
}

// 4. 执行器节点
const executorNode = async (state: typeof PlanningStateAnnotation.State) => {
  console.log(chalk.cyan('--- 执行器 (EXECUTOR): 运行下一步... ---'))
  const plan = state.plan
  const nextStep = plan[0]!

  // 使用健壮的正则来处理单引号和双引号
  const match = nextStep.match(/(\w+)\((?:"|')(.*?)(?:"|')\)/)
  let toolName = 'web_search'
  let query = nextStep

  if (match) {
    toolName = match[1]!
    query = match[2]!
  }

  console.log(
    chalk.blue(
      `--- 执行器 (EXECUTOR): 调用工具 '${toolName}'，查询 '${query}' ---`,
    ),
  )

  const result = await searchTool.invoke({ query })

  // 我们创建 ToolMessage，现在的工具调用是非常安全的。
  const toolMessage = new ToolMessage({
    content: typeof result === 'string' ? result : JSON.stringify(result),
    name: toolName,
    tool_call_id: `manual-${Date.now().toString()}`,
  })

  return {
    plan: plan.slice(1), // 将已经执行的步骤从计划中移除
    intermediate_steps: [toolMessage], // reducer 会将其拼接到数组末尾
  }
}

// 5. 合成器节点
const synthesizerNode = async (state: typeof PlanningStateAnnotation.State) => {
  console.log(chalk.cyan('--- 合成器 (SYNTHESIZER): 生成最终答案... ---'))

  const context = state.intermediate_steps
    .map((msg) => `工具 ${msg.name} 返回: ${msg.content}`)
    .join('\n')

  const prompt = `你是一个专家级的合成器。根据用户的请求和收集到的数据，提供一个全面的最终答案。
  
请求: ${state.user_request}
收集到的数据:
${context}`

  const finalAnswer = await llm.invoke(prompt)
  return { final_answer: finalAnswer.content as string }
}

console.log(chalk.green('计划员、执行器和合成器节点已定义完毕。'))

const planningRouter = (state: typeof PlanningStateAnnotation.State) => {
  if (!state.plan || state.plan.length === 0) {
    console.log(
      chalk.yellow('--- 路由器 (ROUTER): 计划已完成。转到合成器。 ---'),
    )
    return 'synthesize'
  } else {
    console.log(
      chalk.yellow('--- 路由器 (ROUTER): 计划还有更多步骤。继续执行。 ---'),
    )
    return 'execute'
  }
}

const planningGraphBuilder = new StateGraph(PlanningStateAnnotation)
  .addNode('planner', plannerNode)
  .addNode('execute', executorNode)
  .addNode('synthesize', synthesizerNode)
  .addEdge(START, 'planner')
  // 计划之后进行路由判断
  .addConditionalEdges('planner', planningRouter)
  // 执行之后再次进行路由判断
  .addConditionalEdges('execute', planningRouter)
  .addEdge('synthesize', END)

const planningAgentApp = planningGraphBuilder.compile()
console.log(chalk.green('计划型 Agent 已成功编译。'))

const planCentricQuery = `
查找法国、德国和意大利首都的人口。
然后计算它们的总人口。
最后，将该总人口与美国的人口进行比较，说明哪个更大。
`

console.log(
  chalk.bold.green(
    `\n[在同一个查询上测试计划型 Agent]: '${planCentricQuery}'\n`,
  ),
)

// 初始状态输入
const initialPlanningInput = {
  user_request: planCentricQuery,
}

const finalPlanningOutput = await planningAgentApp.invoke(initialPlanningInput)

console.log(chalk.bold.green('\n--- 计划型 Agent 的最终输出 ---'))
console.log(finalPlanningOutput.final_answer)

// -------------------------------------

// Agent 节点，带有一个系统提示词来强制一次仅调用一个工具
const reactAgentNode = async (state: typeof MessagesAnnotation.State) => {
  console.log(chalk.cyan('--- 反应式代理 (REACT): 思考中... ---'))

  const systemPrompt = new SystemMessage(
    '你是一个有用的研究助手。你每次必须只调用一个工具。不要在一次对话中调用多个工具。在收到工具结果后，你再决定下一步。',
  )

  const response = await llmWithTools.invoke([systemPrompt, ...state.messages])
  return { messages: [response] }
}

// 将我们修正后的自定义工具放入 ToolNode
const toolNode = new ToolNode([searchTool])

// 构建带有典型循环特征的 ReAct 图
const reactGraphBuilder = new StateGraph(MessagesAnnotation)
  .addNode('agent', reactAgentNode)
  .addNode('tools', toolNode)
  .addEdge(START, 'agent')
  .addConditionalEdges('agent', toolsCondition)
  .addEdge('tools', 'agent')

const reactAgentApp = reactGraphBuilder.compile()
console.log(chalk.green('反应式 (ReAct) 代理已成功编译。'))

console.log(
  chalk.bold.yellow(
    `\n[在侧重计划的查询上测试反应式 Agent]: '${planCentricQuery}'\n`,
  ),
)

let finalReactOutput: typeof MessagesAnnotation.State | undefined

// 使用 values 模式进行流式传输，以观察每一步的状态更新
const stream = await reactAgentApp.stream(
  { messages: [['user', planCentricQuery]] },
  { streamMode: 'values' },
)

for await (const chunk of stream) {
  finalReactOutput = chunk
  console.log(chalk.bold.magenta('--- 当前状态更新 ---'))
  const lastMessage = chunk.messages[chunk.messages.length - 1]!
  console.log(`[${lastMessage.type}] ${lastMessage.content}`)
  if (AIMessage.isInstance(lastMessage) && lastMessage.tool_calls?.length) {
    console.log(`Tool Calls:`, lastMessage.tool_calls)
  }
  console.log('\n')
}

console.log(chalk.bold.red('\n--- 反应式 Agent 的最终输出 ---'))
if (finalReactOutput) {
  const finalMessage =
    finalReactOutput.messages[finalReactOutput.messages.length - 1]!
  console.log(finalMessage.content)
}

// -------------------------------------

// 定义评估 Agent 解决问题过程的数据模式
const ProcessEvaluationSchema = z.object({
  task_completion_score: z
    .number()
    .describe('1-10分，评价 Agent 是否成功完成了任务。'),
  process_efficiency_score: z
    .number()
    .describe(
      '1-10分，评价 Agent 过程的效率和直接性。分数越高意味着路径越符合逻辑，绕弯路越少。',
    ),
  justification: z.string().describe('对分数的简短证明/理由。'),
})

const judgeLlm = llm.withStructuredOutput(ProcessEvaluationSchema, {
  method: 'jsonMode',
})

const evaluateAgentProcess = async (query: string, finalState: any) => {
  let trace = ''

  // 如果是 ReAct Agent，轨迹保存在 'messages' 中；如果是 Planning，则在 'intermediate_steps'
  if (finalState.messages) {
    trace = finalState.messages
      .map(
        (m) =>
          `${m.type}: ${typeof m.content === 'string' ? m.content : JSON.stringify(m.content)}`,
      )
      .join('\n')
  } else {
    trace = `计划内容: ${JSON.stringify(finalState.original_plan || [])}\n中间步骤: ${JSON.stringify(finalState.intermediate_steps || [])}\n最终答案: ${finalState.final_answer || '未生成'}`
  }

  const prompt = `你是一个人工智能 Agent 的专家评委。请以 1-10 分的等级评估该 Agent 解决任务的过程。
重点关注该过程是否合乎逻辑且高效。
请严格按照以下 JSON 格式返回你的评估结果，不要包含任何其他多余文本：
{
  "task_completion_score": <number>,
  "process_efficiency_score": <number>,
  "justification": <string>
}
    
**用户的任务:** ${query}
**Agent 完整执行轨迹:**
\`\`\`
${trace}
\`\`\`
`
  console.log(chalk.bold(prompt))
  return await judgeLlm.invoke(prompt)
}

console.log(chalk.bold('\n--- 评估反应式 Agent (ReAct) 的过程 ---'))
const reactAgentEvaluation = await evaluateAgentProcess(
  planCentricQuery,
  finalReactOutput,
)
console.dir(reactAgentEvaluation, { depth: null })

console.log(chalk.bold('\n--- 评估计划型 Agent (Planning) 的过程 ---'))
const planningAgentEvaluation = await evaluateAgentProcess(
  planCentricQuery,
  finalPlanningOutput,
)
console.dir(planningAgentEvaluation, { depth: null })
