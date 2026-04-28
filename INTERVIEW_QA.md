# 面试问答：Multi-Agent Debate System

> 适用场景：AI/ML Engineer、Backend Engineer、LLM Application Engineer 岗位面试
> 项目亮点关键词：multi-agent coordination · LangGraph · semantic divergence detection · structured reasoning · auditable AI outputs

---

## 一、项目介绍类（开场必问）

### Q1：能介绍一下这个项目吗？

**考查点：** 你能不能用一句话讲清楚核心价值，而不是把技术栈背一遍。

**面试官想听的方向：** 问题 → 解法 → 亮点，控制在 1 分钟以内。

**示例答案：**

> 单个 LLM 的问题是它很难主动质疑自己的结论——它倾向于"自说自话"。我做了一个多 Agent 辩论系统，让三个认知偏见不同的 Agent（乐观派、悲观派、魔鬼代言人）对同一个问题独立分析，然后基于语义分歧检测让它们互相辩论，最终由第四个 Synthesizer Agent 综合出带置信度的报告。
>
> 核心亮点有两个：一是分歧检测是在 embedding 层面做的，不是简单比较文字；二是置信度分数是代码公式算出来的，不是让 LLM 编一个数字。整个推理过程完全可追溯。

---

### Q2：为什么要做这个项目？解决了什么问题？

**考查点：** 你有没有真正理解 LLM 的局限性，还是只是为了做项目而做项目。

**示例答案：**

> LLM 有一个被研究证实的问题叫 sycophancy（谄媚性）：模型倾向于顺着用户或自己已有的结论走，而不是真正质疑。在需要多角度分析的场景里（比如投资决策、产品评审），这会导致输出看似全面但实际上是一种视角。
>
> 强制用角色各异的 Agent 来辩论，是一种结构性的解决方案——不是靠 prompt 说"请从多角度分析"，而是让不同 Agent 的偏见相互制衡。

---

## 二、技术架构类（重头戏）

### Q3：整体架构是怎样的？

**考查点：** 你对系统各模块的边界是否清晰。

**示例答案：**

> 整个系统是一个单层 LangGraph `StateGraph`，避免了 subgraph 嵌套带来的状态合并复杂性。
>
> 流程：
> 1. `initialize_node` 初始化 debate_id 和状态
> 2. `dispatch_round1` 是一个 routing function，用 `Send` API 把三个 Agent 并行分发出去
> 3. 三个 Agent 并行跑完后，`collect_round1` 做 fan-in，通过 `Annotated[list, add]` reducer 自动聚合
> 4. `divergence_check_node` 调用语义分歧检测，决定是再辩一轮还是去综合
> 5. 如果分歧 > 阈值且未达最大轮次，就再次 fan-out 做 rebuttal；否则进 `synthesize_stub`
> 6. `save_node` 把最终报告写进 SQLite
>
> 关键设计决策：routing function（`dispatch_round1`、`route_divergence`）不作为 node 注册，直接传给 `add_conditional_edges`，这是 LangGraph 1.x 的正确用法，我在开发中实际踩了这个坑并修复了。

---

### Q4：为什么选 LangGraph，而不是 AutoGen 或 CrewAI？

**考查点：** 你对主流框架的横向对比是否有判断，而不是随便选了一个。

**示例答案：**

> 三个框架的核心差异在于控制权。
>
> - **AutoGen** 是对话驱动的，Agent 之间通过消息互相调用，适合聊天场景，但状态管理不够显式，调试困难
> - **CrewAI** 更高层抽象，适合快速搭建，但辩论循环这种需要精确控制"什么时候继续、什么时候停"的逻辑，它的灵活性不够
> - **LangGraph** 给你一个明确的状态机，每个节点的输入/输出都是类型化的，循环是显式的图结构，而且内置 checkpointing 支持回放——这对我需要"完整推理链可追溯"的需求非常关键
>
> 这个项目的核心价值之一就是"auditable"，LangGraph 的 checkpoint 机制天然契合。

---

### Q5：并行 fan-out 是怎么实现的？

**考查点：** 你真正理解 LangGraph 的 Send API，还是只是用了它。

**示例答案：**

> 用的是 LangGraph 的 `Send` API。`dispatch_round1` 函数返回 `list[Send]`，每个 `Send` 指向一个 agent node，payload 里只包含 `topic` 和 `prior_arguments=[]`——这个空列表是 Round 1 隔离性的保证，三个 Agent 互相看不到对方的结论。
>
> ```python
> def dispatch_round1(state: DebateState) -> list[Send]:
>     return [
>         Send("optimist_node", {"topic": state["topic"], "prior_arguments": []}),
>         Send("pessimist_node", {"topic": state["topic"], "prior_arguments": []}),
>         Send("devil_node",     {"topic": state["topic"], "prior_arguments": []}),
>     ]
> ```
>
> 三个 Agent 在同一个 superstep 里并行运行，结果通过 `Annotated[list[AgentArgument], add]` reducer 自动合并到 `current_round_arguments` 字段。这是 LangGraph fan-in 的标准模式。

---

### Q6：分歧检测是怎么做的？为什么不直接比较文本？

**考查点：** 你是否理解 embedding 的局限性，以及为什么要用 key_claims 而不是全文。

**示例答案：**

> 直接比较文本或者对完整 reasoning 做 embedding 会有一个严重问题：所有 Agent 都在讨论同一个 topic，所以它们的文本在语义空间里本来就很接近——即使结论完全相反，余弦相似度也可能很高，导致"误判为收敛"。
>
> 解决方案是只 embed `key_claims`——每个 Agent 提取的 3-7 个短句论点，而不是完整推理文本。论点级别的粒度保留了真正的分歧信号。
>
> 具体实现：用 `BAAI/bge-small-en-v1.5` 模型（本地运行，无额外 API 成本），对每对 Agent 的 key_claims 做 cross-claim 余弦相似度矩阵，取 max similarity，低于 0.75 判定为分歧。必须用 `normalize_embeddings=True`，否则 dot product ≠ cosine similarity，分数会超出 [0,1] 范围。

---

### Q7：置信度分数是怎么计算的？为什么不让 LLM 直接给出？

**考查点：** 这是一个关于 AI 可靠性的核心问题，考查你对 LLM 局限性的理解。

**示例答案：**

> LLM 的 confidence calibration 是臭名昭著的不准确——同一个话题问十次可能给出完全不同的置信度，而且往往和实际的不确定性没有关系。让 LLM 说"我有 85% 的把握"是没有意义的数字。
>
> 我用的是公式：
> ```
> confidence = (1 - max_divergence_score) * round_adjustment
> ```
> 逻辑是：分歧越小说明 Agent 越接近共识，置信度越高；需要的轮次越多说明这个话题越难达成共识，做 0.9/0.8 的折扣。这个数字完全基于辩论过程中可观测的事实，不是 LLM 编的。
>
> `SynthesizerOutput` 这个 Pydantic 模型里故意没有 `confidence_score` 字段，从结构上杜绝了 LLM 输出这个数字的可能性。

---

### Q8：如何防止 Agent "假装"持有自己的立场（sycophancy 问题）？

**考查点：** 这是一个真实的 LLM 工程难题，考查你有没有实际解决过它。

**示例答案：**

> 这是整个项目最大的挑战之一。RLHF 训练让 Claude 倾向于输出平衡、圆滑的答案，简单地说"你是一个悲观主义者"完全没用，Agent 很快就会滑向"但另一方面……"的模式。
>
> 我的解法是方法论驱动的 persona，而不是性格驱动：
> - Optimist：用 VC 投资人视角，具体列 upside scenario 和 catalyst
> - Pessimist：用风险管理师视角，专门找假设漏洞和 downside tail risk
> - Devil's Advocate：专门反驳当前最主流的观点
>
> 每个 prompt 里有明确的 **PROHIBITION 块**，列出禁止出现的词：`however`、`on the other hand`、`balanced view`、`it depends` 等。同时有明确指令：`You must maintain your position unless presented with a logically superior argument. Do not concede to avoid conflict.`
>
> 实际效果：跑了多次测试，三个 Agent 的立场确实有明显差异，不会趋同。

---

## 三、工程设计类

### Q9：如果辩论永远不收敛怎么办？

**考查点：** 你有没有考虑边界情况和系统健壮性。

**示例答案：**

> `route_divergence` 函数里 **第一个检查** 是 max_rounds guard：
> ```python
> if round_num >= max_rounds:
>     return "synthesize_stub"
> ```
> 这个检查必须在分歧检测之前，否则一旦有 bug 导致分歧永远不下降，就会触发 LangGraph 的默认 recursion limit（10007 步），在停下来之前已经发起了几千次 API 调用。
>
> 另外在 `graph.stream` 时显式设置 `recursion_limit=30`，双重保险。Synthesizer 也有 honest non-convergence 路径：当 `convergence_status == "max_rounds"` 时，verdict 会以 "Agents did not reach consensus on this topic." 开头，而不是捏造一个共识。

---

### Q10：为什么 Pydantic 结构化输出会失败？你怎么处理的？

**考查点：** 你对 LLM 结构化输出的实际工程经验。

**示例答案：**

> LLM 偶尔会违反 schema，大约 0.5-1% 的概率。单次调用问题不大，但三个 Agent × 多轮辩论，失败概率就会累积。
>
> 我的处理是使用 `with_structured_output(..., include_raw=True)`，这样解析失败不会直接抛异常，而是在 `result["parsed"]` 里返回 `None`。外层用 retry wrapper 最多重试 2 次，第 3 次失败就注入一个 sentinel `AgentArgument`（`is_sentinel=True, confidence=0.0`），保证图不会 crash，同时在 Synthesizer 里过滤掉 sentinel。
>
> 这个设计让系统在生产环境里有明确的降级路径，而不是随机挂掉。

---

### Q11：Streamlit UI 里实时流式显示是怎么做的？

**考查点：** Streamlit + LangGraph 的集成细节。

**示例答案：**

> 关键是用同步的 `graph.stream(stream_mode="updates")`，**绝对不能用** `astream()` 或 `asyncio.run()`。原因是 Streamlit 的 script runner 线程没有 event loop，而 LangGraph 的同步接口底层用的是 `SyncPregelLoop`，完全不依赖 asyncio，两者天然兼容。
>
> `stream_mode="updates"` 的 chunk 格式是 `{node_name: state_delta}`，每个节点完成时推送一个 chunk。我按 node_name 分发：
> - `optimist_node` / `pessimist_node` / `devil_node` → 渲染 agent 卡片
> - `divergence_check_node` → 显示分歧分数 banner
> - `synthesize_stub` → 渲染最终报告
>
> 注意 `st.rerun()` 只能在 stream 循环结束后调用，在循环内部调用会把 stream 截断。

---

## 四、扩展思考类（加分题）

### Q12：这个系统有什么局限性？你会怎么改进？

**考查点：** 你对自己项目的批判性思维，是否有 production mindset。

**示例答案：**

> 局限性我总结了三个：
>
> 1. **Persona 稳定性依赖 prompt，不稳定。** 目前通过 PROHIBITION 块控制，但没有量化评估 persona drift 的指标。改进方向：加一个 post-call 检查，统计 forbidden phrases 出现频率，超阈值则重试。
>
> 2. **分歧检测阈值是经验值（0.75），没有数据支撑。** 目前的 0.75 是合理起点，但没有在大规模 topic 上做过校准。改进方向：收集 100+ 个 topic 的辩论结果，人工标注是否真的分歧，找最优阈值。
>
> 3. **成本没有做精细控制。** 多轮 × 多 Agent 的 token 消耗增长快。改进方向：按 round 对 prior_arguments 做摘要压缩，而不是传完整历史；用 claude-haiku 做分歧检测的 Claude judge，用 sonnet 做核心分析。
>
> 最想做的扩展：接入真实数据源（财经新闻 API），做一个"分析这支股票值不值得买"的垂直场景，这样置信度分数和辩论报告的价值会更加直观。

---

### Q13：这个系统怎么评估效果好不好？

**考查点：** 你对 LLM 系统评估的理解，这是 AI 工程里的难点。

**示例答案：**

> 这是 LLM 应用的经典难题——没有 ground truth。
>
> 我的思路是分两层：
>
> **过程指标（可量化）：**
> - Persona compliance rate：三个 Agent 的 sentiment 分布是否有显著差异（可以跑 VADER 等情感分析）
> - Concession rate：每轮辩论平均让步几次，太少说明 Agent 没在真正交互，太多说明 sycophancy 没解决
> - Divergence trajectory：分歧分数是否在多轮后下降，下降说明辩论在推动收敛
>
> **结果指标（需要人工）：**
> - 对相同 topic 跑 5 次，看最终报告的 consensus_points 是否稳定（不稳定说明系统噪声大）
> - 和直接问单个 LLM 的回答做对比：辩论系统的输出覆盖了更多视角吗？结论更保守（置信度更低）吗？

---

## 五、HR / 行为类（结尾）

### Q14：这个项目里遇到最难的技术问题是什么？

**考查点：** 你的 problem-solving 过程，以及是否有真实的踩坑经历。

**示例答案：**

> 最难的是 LangGraph 的 Send API 接线方式。文档里有两种写法，但我一开始把 `dispatch_round1` 注册成了 node（`builder.add_node`），然后在那之后用 `add_conditional_edges`，结果 LangGraph 会把 `list[Send]` 的返回值当成 state update dict 来合并，报 `InvalidUpdateError`。
>
> 找这个 bug 花了挺长时间，最后去读了 LangGraph 源码里 `pregel/main.py` 才搞清楚：routing function 必须**直接传给** `add_conditional_edges`，而不是先注册成 node。修法是一行：
> ```python
> # 错误
> builder.add_node("dispatch_round1", dispatch_round1)
> builder.add_conditional_edges("dispatch_round1", lambda s: s)
>
> # 正确
> builder.add_conditional_edges("initialize", dispatch_round1)
> ```
> 这个教训让我对"框架文档和框架实现之间的 gap"更加警惕，重要问题都去翻源码确认。

---

### Q15：如果要把这个系统用在生产环境，你会做什么改造？

**考查点：** 你的 production thinking 和系统设计能力。

**示例答案：**

> 主要四个方向：
>
> 1. **成本控制**：加 token budget per debate，对 round_history 做滚动摘要，Haiku 做廉价的分歧检测 judge，Sonnet 只做核心 Agent 分析
>
> 2. **可观测性**：每次 debate 记录完整 trace（已有 SQLite），加上 latency per node、token usage per agent、persona compliance 指标，接入 Langsmith 或自建 dashboard
>
> 3. **质量保证**：对 Agent 输出做 post-call 自动检查（forbidden phrases 检测、sentiment 分布），不达标的 Agent 输出自动重试，而不是静默通过
>
> 4. **扩展性**：现在 3 个 Agent 是硬编码的，改成配置驱动（可以配置不同数量、不同角色的 Agent），让系统可以适配不同的分析场景（投资、产品评审、法律论证等）

---

*文档维护：有新问题随时补充在对应分类下。*
