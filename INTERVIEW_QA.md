# Multi-Agent Debate System 面试题库

> 适用岗位：AI/ML Engineer、LLM Application Engineer、Backend Engineer
> 关键词：multi-agent coordination · LangGraph · semantic divergence detection · structured reasoning · auditable AI outputs

---

## Q1：为什么要用多个 Agent 互相辩论，而不是直接让一个 LLM 从多角度分析？

*（也可能被问成：单 LLM 做多角度分析有什么问题？你的系统解决了什么？）*

### 面试官想听到的
考察点：**你对 LLM 核心局限性的理解**，以及架构决策背后的 trade-off，而不只是描述"我用了多个 Agent"。

### 代码中的实际方案
`debate/prompts.py` 中每个 Agent 有独立的 system prompt，各自定义了不同的分析方法论：
```python
AGENT_PROMPTS = {
    "optimist": """You are a venture capital investment analyst...
        ANALYTICAL FRAMEWORK:
        1. Identify the primary value driver...
        PROHIBITION: Never use these phrases: "however", "on the other hand",
        "balanced view", "it depends", "while there are risks"
        ...You must maintain your position unless presented with a logically
        superior argument. Do not concede to avoid conflict.""",
    "pessimist": """You are a risk management specialist...""",
    "devil": """You are a strategic consultant specializing in contrarian analysis..."""
}
```

Round 1 通过 `Send` API 强制隔离：
```python
def dispatch_round1(state: DebateState) -> list[Send]:
    return [
        Send("optimist_node", {"topic": state["topic"], "prior_arguments": []}),
        ...
    ]
```
`prior_arguments=[]` 保证三个 Agent 在 Round 1 互相看不到结论。

### 如何对面试官表述
> "单 LLM 有一个被研究证实的问题叫 sycophancy——模型倾向于顺着已有结论走，即使你在 prompt 里说'请从多角度分析'，它也会给出表面上平衡但实际上是同一视角的内容。
>
> 我的解法是结构性的，不是 prompt 层面的：用三个有不同认知偏见的 Agent，在 Round 1 完全隔离各自独立分析，然后强制让它们互相质疑。每个 Agent 的 system prompt 里有明确的 PROHIBITION 块，禁止出现'however'、'on the other hand'这类妥协性语言，并且明确要求'除非遇到逻辑上更优的论据，否则必须坚持立场'。
>
> 单 LLM 的问题是它无法真正质疑自己；多 Agent 的设计让质疑变成了结构性的必然。"

### 亮点
- 能说出 sycophancy 这个具体问题，而不是泛泛说"多角度"
- PROHIBITION 块是工程化的 sycophancy 防御，不只是 prompt 说说

### 瓶颈
- Persona 稳定性依赖 prompt，没有量化评估 persona drift 的指标
- RLHF 训练的模型天然倾向平衡，PROHIBITION 只能减轻，不能根治

### 突出的能力
**对 LLM 核心局限性的理解** + **结构性问题的工程化解决思路**

---

## Q2：LangGraph 的 Send API 是怎么实现并行 fan-out 的？你踩过什么坑？

*（也可能被问成：三个 Agent 怎么并行跑的？LangGraph 的状态合并是怎么做的？）*

### 面试官想听到的
考察点：**LangGraph 的实际使用经验**，以及能否说清楚 fan-out/fan-in 的具体实现机制，而不只是说"我用了 LangGraph"。

### 代码中的实际方案
`debate/nodes/dispatch.py`：
```python
from langgraph.types import Send

def dispatch_round1(state: DebateState) -> list[Send]:
    return [
        Send("optimist_node", {"topic": state["topic"], "prior_arguments": []}),
        Send("pessimist_node", {"topic": state["topic"], "prior_arguments": []}),
        Send("devil_node",     {"topic": state["topic"], "prior_arguments": []}),
    ]
```

`debate/state.py` 中的 fan-in reducer：
```python
class DebateState(TypedDict, total=False):
    current_round_arguments: Annotated[list[AgentArgument], add]
    # 只有这一个字段用 add reducer，其余都是 last-write-wins
```

`debate/graph.py` 中的关键接线方式：
```python
# 正确：routing function 直接传给 add_conditional_edges
builder.add_conditional_edges("initialize", dispatch_round1)

# 错误（踩过的坑）：
# builder.add_node("dispatch_round1", dispatch_round1)  ← 不能注册为 node
# builder.add_conditional_edges("dispatch_round1", lambda s: s)
```

### 如何对面试官表述
> "Send API 的工作方式是：routing function 返回一个 `list[Send]`，每个 Send 指向一个 node 和它的 payload。LangGraph 会在同一个 superstep 里并行执行所有 Send，结果通过 `Annotated[list[AgentArgument], add]` reducer 自动合并到状态里。
>
> 我踩了一个坑：一开始把 `dispatch_round1` 注册成了 node，然后在它之后用 `add_conditional_edges`，结果 LangGraph 会把 `list[Send]` 的返回值当成 state update dict 来合并，报 `InvalidUpdateError`。正确用法是把 routing function 直接传给 `add_conditional_edges`，不注册成 node。这个坑去读了 LangGraph 源码 `pregel/main.py` 才搞清楚。"

### 亮点
- 能说清楚 Send API + add reducer 的完整机制
- 有真实踩坑经历，说明是实际写过代码的

### 瓶颈
- `add` reducer 是追加语义，如果同一个 Agent 因为 retry 跑了两次，会有重复数据，需要在 collect 节点里处理
- 并行 Agent 的执行顺序不确定，UI 渲染时 chunk 到达顺序也不确定

### 突出的能力
**LangGraph 的实际工程经验** + **从踩坑到读源码的问题排查能力**

---

## Q3：分歧检测为什么不直接比较 Agent 的完整回答，而是用 key_claims 做 embedding？

*（也可能被问成：你的分歧检测是怎么做的？为什么选这个方案？）*

### 面试官想听到的
考察点：**对 embedding 语义空间的理解**，以及为什么 naive 方案会失效。

### 代码中的实际方案
`debate/divergence.py`：
```python
def compute_divergence(arguments: list[AgentArgument]) -> tuple[float, list[tuple[str, str]]]:
    for arg_a, arg_b in combinations(arguments, 2):
        # 只 embed key_claims，不 embed 完整 reasoning
        all_claims = arg_a.key_claims + arg_b.key_claims
        # CRITICAL: normalize_embeddings=True 才能保证 dot product == cosine similarity
        embeddings = model.encode(all_claims, normalize_embeddings=True)
        emb_a = embeddings[:len(arg_a.key_claims)]
        emb_b = embeddings[len(arg_a.key_claims):]
        # cross-claim 相似度矩阵，取最大值
        sim_matrix = emb_a @ emb_b.T
        max_sim = float(sim_matrix.max())
```

`debate/state.py` 中 `AgentArgument` 的 `key_claims` 字段：
```python
key_claims: list[str] = Field(
    min_length=3,
    description="3–7 short extractable claims (used for embedding in Phase 2)",
)
```

### 如何对面试官表述
> "直接对完整 reasoning 做 embedding 有一个根本性问题：三个 Agent 讨论的是同一个 topic，所以它们的文本在语义空间里本来就很接近——即使乐观派说'远程工作提升生产力'，悲观派说'远程工作破坏协作'，两段文字里都有'远程工作'、'生产力'、'协作'这些词，embedding 相似度会很高，误判为已经收敛。
>
> 解法是只 embed `key_claims`——每个 Agent 提取的 3-7 个短句核心论点，而不是完整推理文本。论点级别的粒度保留了真正的分歧信号。
>
> 另外有个技术细节：必须用 `normalize_embeddings=True`，否则 dot product 不等于 cosine similarity，分数会超出 [0,1] 范围，阈值比较就乱了。"

### 亮点
- 能说清楚 naive 方案（全文 embedding）失效的根本原因
- `normalize_embeddings=True` 这个细节说明真正写过代码

### 瓶颈
- 阈值 0.75 是经验值，没有在大规模 topic 上做过校准
- 没有 Claude judge 做二次验证，borderline 区间（0.75-0.97）全部保守判定为分歧

### 突出的能力
**对 embedding 语义空间的深入理解** + **工程细节的扎实程度**

---

**追问：DIVERGE_THRESHOLD 是 0.75，这个值是怎么来的？如果调高或调低会怎样？**

`debate/divergence.py` 中：
```python
DIVERGE_THRESHOLD: float = 0.75
CONVERGE_FAST_PATH: float = 0.97
```

0.75 是合理起点，但没有数据支撑。

- **调高（比如 0.85）**：更容易判定为收敛，辩论轮次减少，成本降低，但可能在真正有分歧时提前终止
- **调低（比如 0.60）**：更容易判定为分歧，辩论轮次增加，分析更充分，但成本更高，可能在已经收敛后还继续辩论

**理想的校准方案**：收集 50-100 个 topic 的辩论结果，人工标注"这轮是否真的有分歧"，用这个 ground truth 找最优阈值（最大化 F1 score）。

**对面试官表述：**
> "0.75 是基于直觉的起点，没有系统校准过。理想方案是收集一批 topic，人工标注每轮是否真的有分歧，用 ground truth 找最优阈值。调高阈值会提前终止辩论，调低会增加轮次和成本，本质是 precision 和 recall 的 trade-off。"

**突出的能力：**
**对超参数校准的数据驱动意识** + **对 trade-off 的清晰表达**

---

## Q4：置信度分数为什么不让 LLM 直接给出？你是怎么设计的？

*（也可能被问成：confidence score 是怎么计算的？为什么不信任 LLM 的自我评估？）*

### 面试官想听到的
考察点：**对 LLM 可靠性局限的理解**，以及如何用工程手段保证输出的可信度。

### 代码中的实际方案
`debate/nodes/synthesize.py`：
```python
class SynthesizerOutput(BaseModel):
    """
    confidence_score is intentionally absent. It is computed in Python
    by _compute_confidence_score() after this call. Including it here
    would allow the LLM to invent a number, violating SYNTH-03.
    """
    consensus_points: list[str]
    disputed_points: list[DisputedPoint]
    verdict: str
    # 故意没有 confidence_score 字段

_ROUND_ADJUSTMENTS: dict[int, float] = {1: 1.0, 2: 0.9, 3: 0.8}

def _compute_confidence_score(round_history: list[RoundRecord], round_num: int) -> float:
    # 从所有轮次的历史中取最大分歧分数
    max_divergence = max(
        (r.divergence_score for r in round_history), default=0.0
    )
    raw = 1.0 - max_divergence
    adjustment = _ROUND_ADJUSTMENTS.get(round_num, 0.8)
    return round(raw * adjustment, 4)
```

### 如何对面试官表述
> "LLM 的 confidence calibration 是出了名的不准确——同一个 topic 问十次可能给出完全不同的置信度，而且和实际的不确定性没有关系。让 LLM 说'我有 85% 的把握'是没有意义的数字。
>
> 我的方案是从 `SynthesizerOutput` 这个 Pydantic 模型里故意去掉 `confidence_score` 字段，从结构上杜绝 LLM 输出这个数字的可能性，然后用公式在 Python 里算：`(1 - max_divergence_score) * round_adjustment`。
>
> 逻辑是：分歧越小说明 Agent 越接近共识，置信度越高；需要的轮次越多说明 topic 越难达成共识，做 0.9/0.8 的折扣。这个数字完全基于辩论过程中可观测的事实，不是 LLM 编的。"

### 亮点
- 从 Pydantic schema 层面结构性禁止，不只是 prompt 说"不要给数字"
- 公式有清晰的业务语义，可解释

### 瓶颈
- 公式简单，没有考虑 Agent 让步数量等其他信号
- `round_adjustment` 的具体数值（0.9, 0.8）是经验值

### 突出的能力
**对 LLM 可靠性问题的工程化解决** + **可解释 AI 的设计意识**

---

## Q5：辩论循环怎么保证不会无限跑下去？stop condition 是怎么设计的？

*（也可能被问成：什么时候停止辩论？如果 Agent 永远不收敛怎么办？）*

### 面试官想听到的
考察点：**系统健壮性设计**，能否说清楚边界情况处理，以及 stop condition 的优先级。

### 代码中的实际方案
`debate/nodes/dispatch.py` 中的 `route_divergence`：
```python
def route_divergence(state: DebateState) -> list[Send] | str:
    round_num = state.get("round_num", 0)
    max_rounds = state.get("max_rounds", 3)
    divergence_score = state.get("divergence_score", 0.0)

    # Guard 1: 硬上限——必须在检查分歧之前
    if round_num >= max_rounds:
        return "synthesize_stub"

    # Guard 2: 收敛——分歧低于阈值
    if divergence_score < DIVERGE_THRESHOLD:
        return "synthesize_stub"

    # 继续辩论
    return _build_rebuttal_sends(state)
```

`debate/graph.py` 中：
```python
graph.stream(
    ...,
    config={"recursion_limit": 30}  # 双重保险
)
```

`debate/nodes/synthesize.py` 中的 honest non-convergence 路径：
```python
if convergence_status == "max_rounds":
    convergence_note = (
        "\n\nIMPORTANT: ... Your verdict MUST begin with exactly: "
        "'Agents did not reach consensus on this topic.'"
    )
```

### 如何对面试官表述
> "有两层保护。第一层是 `route_divergence` 里的 max_rounds guard，关键是它必须是第一个检查，在分歧检测之前。如果反过来，一旦有 bug 导致分歧永远不下降，就会触发 LangGraph 的默认 recursion limit（10007 步），在停下来之前已经发起了几千次 API 调用。第二层是 `graph.stream` 时显式设置 `recursion_limit=30`，双重兜底。
>
> 另外 Synthesizer 有 honest non-convergence 路径：如果是因为达到最大轮次而停止，verdict 必须以'Agents did not reach consensus on this topic.'开头，不捏造共识。"

### 亮点
- max_rounds guard 的顺序优先级有明确的工程理由
- non-convergence 的诚实路径，体现了对 AI 可信度的重视

### 瓶颈
- max_rounds 默认值 3 是经验值，不同 topic 复杂度差异大
- `recursion_limit=30` 是硬编码，应该可配置

### 突出的能力
**系统健壮性设计** + **边界情况的防御性思维**

---

## Q6：Pydantic 结构化输出失败了怎么处理？你是怎么设计 retry 机制的？

*（也可能被问成：LLM 输出不符合 schema 怎么办？如何保证图不会 crash？）*

### 面试官想听到的
考察点：**LLM 工程的实际经验**，能否说清楚结构化输出失败的处理链路和降级策略。

### 代码中的实际方案
`debate/nodes/agents.py` 中的 `_invoke_with_retry`：
```python
def _invoke_with_retry(
    llm_chain,
    input_data: dict,
    agent_role: str,
    round_num: int,
    max_retries: int = 2,
) -> AgentArgument:
    # include_raw=True: 解析失败时 result["parsed"] 是 None，不直接抛异常
    structured_llm = llm.with_structured_output(AgentArgument, include_raw=True)

    for attempt in range(max_retries + 1):
        try:
            result = structured_llm.invoke(input_data)
            if result.get("parsed") is not None:
                return result["parsed"]
        except Exception:
            pass

    # 第三次失败：注入 sentinel，保证图不 crash
    return AgentArgument(
        agent_role=agent_role,
        round_num=round_num,
        position="[Analysis unavailable due to parsing error]",
        reasoning="",
        confidence=0.0,
        key_claims=[],
        is_sentinel=True,
    )
```

`debate/nodes/synthesize.py` 中过滤 sentinel：
```python
def _build_synthesis_context(state: DebateState) -> str:
    for arg in round_record.arguments:
        if arg.is_sentinel:
            lines.append(f"[{arg.agent_role.upper()}] <no data — sentinel>")
            continue
```

### 如何对面试官表述
> "LLM 大约有 0.5-1% 的概率违反 schema，单次问题不大，但三个 Agent 乘以多轮，失败概率会累积。
>
> 关键是用 `include_raw=True`，这样解析失败不会直接抛异常，而是在 `result['parsed']` 里返回 None，外层可以 catch 并 retry。最多重试 2 次，第 3 次失败就注入一个 sentinel `AgentArgument`（`is_sentinel=True, confidence=0.0`），保证图不 crash。
>
> Synthesizer 里会过滤掉 sentinel，不让它影响最终报告。这样整个系统有明确的降级路径，而不是随机挂掉。"

### 亮点
- `include_raw=True` 是 langchain 的高级用法，说明真正研究过 API
- sentinel 模式保证了图的完整性，降级而不是崩溃

### 瓶颈
- sentinel 会让最终报告缺少一个 Agent 的视角，质量下降
- 没有对 retry 原因分类（是 schema 违反还是网络错误），无法针对性优化

### 突出的能力
**LLM 工程的实际经验** + **降级策略的分层设计**

---

## Q7：Rebuttal 轮次里 Agent 怎么知道其他人说了什么？你怎么控制 context 长度？

*（也可能被问成：多轮辩论的上下文是怎么传递的？有没有 token 爆炸的风险？）*

### 面试官想听到的
考察点：**多轮对话的上下文管理**，以及 token 成本控制意识。

### 代码中的实际方案
`debate/nodes/dispatch.py` 中的 `_build_compact_summaries`：
```python
def _build_compact_summaries(
    arguments: list[AgentArgument],
    exclude_role: str
) -> list[str]:
    summaries = []
    for arg in arguments:
        if arg.agent_role == exclude_role or arg.is_sentinel:
            continue
        # 只传 position + top 3 key_claims + confidence，不传完整 reasoning
        claims_text = "\n".join(f"  - {c}" for c in arg.key_claims[:3])
        summary = (
            f"[{arg.agent_role.upper()}]\n"
            f"Position: {arg.position}\n"
            f"Key claims:\n{claims_text}\n"
            f"Confidence: {arg.confidence:.0%}"
        )
        summaries.append(summary)
    return summaries
```

每个 Agent 在 rebuttal 轮次收到的 payload：
```python
Send("optimist_node", {
    "topic": state["topic"],
    "prior_arguments": compact_summaries,  # ~100 词，不是完整历史
    "round_num": current_round,
})
```

### 如何对面试官表述
> "每个 Agent 在 rebuttal 轮次收到的不是完整的对话历史，而是其他 Agent 的 compact summary：position 一句话 + top 3 key_claims + confidence，大约 100 个词。完整的 reasoning 不传，因为它很长而且对反驳来说信息密度低。
>
> 这样做有两个好处：一是控制 token 消耗，不会随轮次线性增长；二是强迫 Agent 聚焦在论点上，而不是被对方的措辞带偏。
>
> 另外每个 Agent 只收到其他人的 summary，不收到自己的，这是通过 `exclude_role` 参数实现的。"

### 亮点
- compact summary 的设计同时解决了 token 和质量两个问题
- `exclude_role` 的细节说明对 Agent 的信息隔离有清晰的设计意图

### 瓶颈
- 只传 top 3 key_claims 可能丢失重要论据
- 没有跨轮次的 summary 压缩，如果 max_rounds 很大，每轮还是会重新发 compact summary

### 突出的能力
**多轮对话的上下文管理** + **token 成本控制意识**

---

## Q8：让步（concession）机制是怎么设计的？Agent 在什么情况下会让步？

*（也可能被问成：Agent 怎么知道要让步？concession 的数据结构是怎么设计的？）*

### 面试官想听到的
考察点：**结构化输出的设计能力**，以及如何让 LLM 的行为可追溯。

### 代码中的实际方案
`debate/state.py` 中的 `Concession` 模型：
```python
class Concession(BaseModel):
    conceded_point: str      # 让步的具体论点
    triggered_by_agent: str  # 谁的论据触发了这次让步
    triggered_by_claim: str  # 具体是哪句话说服了我
    rationale: str           # 一句话解释为什么让步
```

`debate/nodes/agents.py` 中 rebuttal 轮次的 prompt 附加：
```python
if round_num > 0:
    rebuttal_instructions = """
    --- Rebuttal instructions ---
    If you concede a point, populate the concessions field:
    - conceded_point: the specific claim you yield
    - triggered_by_agent: which agent made the argument (e.g. "pessimist")
    - triggered_by_claim: copy the EXACT claim text that convinced you
    - rationale: one sentence explaining why you concede

    Do NOT concede to avoid conflict or to appear balanced.
    Only concede on logical grounds.
    """
```

### 如何对面试官表述
> "让步机制的核心是 `Concession` 这个 Pydantic 模型的设计：它要求 Agent 不只说'我让步了'，还要说是谁的哪句话说服了它（`triggered_by_agent` + `triggered_by_claim`），以及为什么（`rationale`）。这样每次让步都有完整的因果链，最终报告里的 concession_log 是完全可追溯的。
>
> 在 rebuttal 轮次的 prompt 里，明确要求'只有在逻辑上更优的论据出现时才让步，不要为了显得平衡而让步'，这是对 sycophancy 的第二道防线。"

### 亮点
- `triggered_by_claim` 要求复制原文，强制精确溯源，不能模糊说"对方说的有道理"
- 让步指令和 PROHIBITION 块形成互补：前者防止不该让步，后者规范让步的格式

### 瓶颈
- LLM 可能随意填写 `triggered_by_claim`，不一定真的是对方说过的话
- 没有验证 `triggered_by_claim` 是否真的出现在对方的 key_claims 里

### 突出的能力
**结构化输出设计** + **AI 输出可追溯性的工程实现**

---

## Q9：Streamlit UI 里的实时流式显示是怎么做的？为什么不用 async？

*（也可能被问成：用户怎么看到 Agent 一个个出来的？streaming 是怎么实现的？）*

### 面试官想听到的
考察点：**Streamlit + LangGraph 集成的技术细节**，以及对 async/sync 的正确理解。

### 代码中的实际方案
`app.py` 中的 streaming 循环：
```python
for chunk in graph.stream(
    {"topic": topic, "max_rounds": max_rounds},
    config={"configurable": {"thread_id": thread_id}, "recursion_limit": 30},
    stream_mode="updates"  # 每个 node 完成时推送一个 chunk
):
    for node_name, node_update in chunk.items():
        _render_agent_chunk(node_name, node_update)
```

`_render_agent_chunk` 按 node_name 分发：
```python
AGENT_NODES = {"optimist_node", "pessimist_node", "devil_node"}

def _render_agent_chunk(node_name: str, node_update: dict):
    if node_name in AGENT_NODES:
        args = node_update.get("current_round_arguments", [])
        for arg in args:
            with st.expander(f"{label} — Round {arg.round_num + 1}", expanded=True):
                st.markdown(f"**Position:** {arg.position}")
                st.progress(arg.confidence, text=f"Confidence: {arg.confidence:.0%}")
```

### 如何对面试官表述
> "用的是同步的 `graph.stream(stream_mode='updates')`，绝对不能用 `astream()` 或 `asyncio.run()`。原因是 Streamlit 的 script runner 线程没有 event loop，而 LangGraph 的同步接口底层用的是 `SyncPregelLoop`，完全不依赖 asyncio，两者天然兼容。用 async 会报 `RuntimeError: no running event loop`。
>
> `stream_mode='updates'` 的 chunk 格式是 `{node_name: state_delta}`，每个 node 完成时推送一个 chunk。我按 node_name 分发渲染：agent node 来了渲染 Agent 卡片，divergence_check_node 来了显示分歧分数，synthesize_stub 来了渲染最终报告。
>
> 还有一个细节：`st.rerun()` 只能在 stream 循环结束后调用，在循环里调用会把 stream 截断。"

### 亮点
- 能说清楚 sync 的原因，而不只是说"我用了 sync"
- `stream_mode="updates"` 的 chunk 格式理解准确

### 瓶颈
- parallel Send fan-out 的三个 Agent chunk 到达顺序不确定，UI 渲染顺序可能不稳定
- Streamlit 没有 WebSocket，每次 st.rerun 都是完整的页面重渲染

### 突出的能力
**Streamlit + LangGraph 集成的实际经验** + **async/sync 机制的深入理解**

---

## Q10：SQLite 持久化是怎么设计的？save_node 为什么不修改 state？

*（也可能被问成：辩论结果存在哪里？怎么实现历史辩论回放的？）*

### 面试官想听到的
考察点：**副作用节点的设计原则**，以及 LangGraph 状态机中 side-effect 的处理方式。

### 代码中的实际方案
`debate/store.py`：
```python
def save_debate(report: DebateReport, conn: sqlite3.Connection | None = None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO debates (debate_id, topic, created_at, status, report_json) VALUES (?,?,?,?,?)",
        (report.debate_id, report.topic, ..., report.model_dump_json())
    )
    conn.commit()

def load_debate(debate_id: str, ...) -> DebateReport | None:
    row = conn.execute("SELECT report_json FROM debates WHERE debate_id = ?", (debate_id,)).fetchone()
    if row is None:
        return None
    return DebateReport.model_validate_json(row["report_json"])
```

`debate/nodes/save.py`：
```python
def save_node(state: DebateState) -> dict:
    report = state.get("final_report")
    if report is not None:
        save_debate(report)
    return {}  # 返回空 dict，不修改任何 state 字段
```

### 如何对面试官表述
> "save_node 的设计原则是：它是一个纯副作用节点，只做 SQLite 写入，不修改任何 state 字段，所以返回空 dict。这样做有两个好处：一是 state 的不可变性，save_node 不会影响后续节点的行为；二是如果 save 失败，静默处理不 crash，下次重跑可以通过 `INSERT OR REPLACE` 幂等写入。
>
> 回放用 `DebateReport.model_validate_json()` 从 JSON 直接反序列化，不需要重跑任何 Agent，完全从 DB 恢复。"

### 亮点
- 副作用节点返回空 dict 是 LangGraph 的正确模式，说明理解状态机设计原则
- `INSERT OR REPLACE` 保证幂等性，多次 save 不会报错

### 瓶颈
- 没有 save 失败的监控，静默处理可能导致数据丢失
- Pydantic 的 `model_dump_json()` 对大型 DebateReport 可能产生很大的 JSON

### 突出的能力
**副作用节点的设计原则** + **幂等性意识**

---

## Q11：这个系统有什么局限性？如果要上生产你会做哪些改造？

*（也可能被问成：你觉得这个项目哪里做得不够好？下一步怎么优化？）*

### 面试官想听到的
考察点：**批判性思维和 production mindset**，能否客观评估自己的系统，而不是只说好话。

### 代码中的实际方案
结合代码中可见的局限：
1. `DIVERGE_THRESHOLD = 0.75`：硬编码经验值，无数据支撑
2. `_ROUND_ADJUSTMENTS = {1: 1.0, 2: 0.9, 3: 0.8}`：同上
3. 没有 token 使用监控（无 Prometheus / 无 usage logging）
4. 没有 prompt versioning（prompt 改了不知道哪个版本出了问题）
5. `save_node` 失败静默处理，无告警

### 如何对面试官表述
> "我总结了三个主要局限：
>
> 第一，**Persona 稳定性没有量化**。现在靠 PROHIBITION 块控制 sycophancy，但没有自动化检测，不知道实际有多少比例的输出违反了 persona 约束。改进方向：加一个 post-call 检查，统计 forbidden phrases 出现频率，超阈值就重试或告警。
>
> 第二，**分歧阈值是经验值**。0.75 合理但没有数据支撑。改进方向：收集 100+ topic 的辩论结果，人工标注，找最优阈值。
>
> 第三，**成本没有精细控制**。多轮多 Agent 的 token 消耗增长快，但没有监控。改进方向：加 token usage logging，按 Agent 和 round 分维度统计；实现 prompt caching，重复的 system prompt 部分缓存复用。
>
> 如果要上生产，还需要加：完整的可观测性（trace per debate、latency per node）、rate limiting、以及 Haiku 做分歧检测的 Claude judge 来降低成本。"

### 亮点
- 局限性说得具体，有代码支撑，不是泛泛而谈
- 每个局限都配了具体的改进方向

### 瓶颈
- 改进方向都是增量的，没有提到架构层面的根本性改造

### 突出的能力
**自我批判能力** + **从 demo 到 production 的工程思维**

---

## Q12：怎么评估这个系统的效果好不好？你怎么知道辩论结果是有价值的？

*（也可能被问成：有没有做过评测？怎么量化辩论质量？）*

### 面试官想听到的
考察点：**AI 系统评测的方法论**，这是 LLM 应用工程的难点，能说清楚说明有深度。

### 代码中的实际方案
代码里没有系统化的评估体系，只有基础的功能测试（`tests/test_phase2.py` 等）。

### 如何对面试官表述
> "这是 LLM 应用的经典难题——没有 ground truth。我的思路是分两层：
>
> **过程指标（可量化）：**
> - Persona compliance rate：三个 Agent 的 sentiment 分布是否有显著差异，可以用 VADER 等情感分析工具跑
> - Concession rate：每轮平均让步几次，太少说明 Agent 没在真正交互，太多说明 sycophancy 没解决
> - Divergence trajectory：分歧分数是否在多轮后下降，下降说明辩论在推动收敛
>
> **结果指标（需要人工）：**
> - 对相同 topic 跑 5 次，看 consensus_points 是否稳定，不稳定说明系统噪声大
> - 和直接问单个 LLM 的回答做对比：辩论系统的输出覆盖了更多视角吗？结论更保守（置信度更低）吗？
>
> 目前这个项目还没有系统化评估，是明显的 gap。"

### 亮点
- 主动承认评估 gap，而不是回避
- 过程指标 + 结果指标的分层，说明对 LLM 评估有方法论

### 突出的能力
**AI 系统评测的方法论** + **诚实面对系统局限的态度**

---

## Q13：如果要给这个系统加一个"菜单分析"的垂直场景，你会怎么设计？

*（也可能被问成：这个系统能用在哪些实际场景？举一个具体的例子说明。）*

### 面试官想听到的
考察点：**系统扩展性思维**，以及能否把通用架构映射到具体业务场景。

### 如何对面试官表述
> "以菜单分析为例，比如'这道菜值不值得推荐给素食用户'。
>
> 现有架构基本不用改，只需要三个层面的定制：
>
> 第一，**Domain-specific personas**：把 Optimist 改成'营养师视角'（关注健康价值），Pessimist 改成'过敏风险专家'（关注食材风险），Devil's Advocate 改成'价格分析师'（关注性价比）。
>
> 第二，**接入数据源**：给 Agent 加 tool use，能查菜品成分数据库、用户历史订单、过敏原数据。这样 Agent 的论点基于真实数据，不只是 LLM 的先验知识。
>
> 第三，**输出格式适配**：`DebateReport` 的 `disputed_points` 可以直接映射到'哪些用户群体适合/不适合'，`confidence_score` 可以作为推荐置信度。
>
> 整个 LangGraph 图、分歧检测、置信度公式都不需要改，这说明架构的通用性是真实的。"

### 亮点
- 说出了具体需要改什么、不需要改什么，说明对架构有清晰认知
- tool use 的设计让 Agent 有了真实数据支撑，不只是 LLM 幻想

### 突出的能力
**系统扩展性设计** + **通用架构到垂直场景的映射能力**

---

---

## Q14：`collect_round1` 这个节点在 Round 1 和 Rebuttal 轮次里都被复用，它是怎么做到的？有什么风险？

*（也可能被问成：多轮辩论的 fan-in 是怎么实现的？为什么不给 rebuttal 单独写一个 collect 节点？）*

### 面试官想听到的
考察点：**LangGraph 状态机的节点复用设计**，能否说清楚为什么能复用以及复用带来的约束。

### 代码中的实际方案
`debate/nodes/collect.py`：
```python
def collect_round1(state: DebateState) -> dict:
    current_args = state.get("current_round_arguments", [])
    round_num = state.get("round_num", 0)
    round_history = state.get("round_history", [])

    new_record = RoundRecord(
        round_num=round_num,
        arguments=current_args,
        divergence_score=0.0,  # 由 divergence_check_node 后续回填
    )
    return {
        "round_history": round_history + [new_record],
        "current_round_arguments": [],   # 重置累加器
        "round_num": round_num + 1,
        "status": "running",
    }
```

`debate/graph.py` 中，所有 agent node（Round 1 和 rebuttal 轮）都指向同一个节点：
```python
for role in ["optimist_node", "pessimist_node", "devil_node"]:
    builder.add_edge(role, "collect_round1")
```

### 如何对面试官表述
> "能复用的根本原因是 `collect_round1` 的逻辑完全基于 state 字段，和它是第几轮无关：读取 `current_round_arguments`（由 `add` reducer 自动聚合好的），打包成 `RoundRecord`，追加到 `round_history`，然后把 `current_round_arguments` 清空。这个操作在 Round 1 和 Round 2、3 里语义完全一样。
>
> 复用的好处是图的拓扑更简洁，不需要额外的 collect_rebuttal 节点。但有一个隐藏约束：`current_round_arguments` 的 `add` reducer 是追加语义，如果同一轮的某个 agent 因为 retry 被调用了两次，就会出现重复 AgentArgument。这种情况下 collect_round1 会把重复数据都打包进 RoundRecord，synthesizer 看到的就是 4 个 argument 而不是 3 个。
>
> 目前代码里没有去重逻辑，是一个已知的 gap。"

### 亮点
- 说清楚复用的前提条件（逻辑与轮次无关）
- 主动指出 add reducer 的追加语义带来的重复风险

### 瓶颈
- 没有去重保护，retry 场景下 RoundRecord.arguments 可能超过 3 个
- `divergence_score=0.0` 是占位值，真正的值要等 divergence_check_node 回填，中间有一段时间 RoundRecord 的 divergence_score 是错的

### 突出的能力
**LangGraph 节点复用的设计意识** + **状态机副作用的边界分析**

---

## Q15：`initialize_node` 里 `max_rounds` 有默认值 3，但如果调用方传了 `max_rounds=0` 会怎样？

*（也可能被问成：图的输入参数有没有做校验？边界情况是怎么处理的？）*

### 面试官想听到的
考察点：**防御性编程意识**，能否主动发现输入校验的缺失，而不是只描述正常路径。

### 代码中的实际方案
`debate/nodes/initialize.py`：
```python
def initialize_node(state: DebateState) -> dict:
    return {
        "debate_id": str(uuid.uuid4()),
        "round_num": 0,
        "max_rounds": state.get("max_rounds", 3),  # 没有 > 0 的校验
        ...
    }
```

`debate/nodes/dispatch.py` 中的 `route_divergence`：
```python
if round_num >= max_rounds:   # 0 >= 0 → True，第一轮就直接终止
    return "synthesize_stub"
```

### 如何对面试官表述
> "如果传 `max_rounds=0`，`initialize_node` 会直接存入 0，没有任何校验。然后 Round 1 的三个 Agent 跑完，进入 `divergence_check_node`，`route_divergence` 检查 `round_num >= max_rounds`，此时 `round_num=1`（collect_round1 已经 +1），`max_rounds=0`，条件成立，直接跳到 synthesize。
>
> 结果是：系统会完成 Round 1 的分析，然后直接综合，不会发生 rebuttal。这个行为不是崩溃，但也不是用户预期的——传 0 可能意味着'我不想辩论'，也可能是输入错误。
>
> 正确的修法是在 initialize_node 里加：`max_rounds = max(1, state.get('max_rounds', 3))`，或者在 Streamlit 的 slider 层面限制最小值为 1（代码里 slider 已经设置了 `min_value=1`，所以 UI 路径是安全的，但直接调用 `graph.invoke` 的路径没有保护）。"

### 亮点
- 能追踪 `max_rounds=0` 的完整执行路径，说清楚结果是什么
- 区分了 UI 路径（有保护）和直接调用路径（无保护）

### 瓶颈
- 这个 bug 在生产环境里影响有限（UI 限制了 min=1），但 API 直接调用时是真实风险

### 突出的能力
**输入边界的防御性分析** + **完整执行路径的追踪能力**

---

## Q16：`DebateState` 用了 `TypedDict` 而不是 Pydantic `BaseModel`，为什么？两者在 LangGraph 里有什么区别？

*（也可能被问成：图的状态为什么不用 Pydantic？TypedDict 的 `total=False` 是什么意思？）*

### 面试官想听到的
考察点：**LangGraph 状态设计的技术选型**，以及 TypedDict 和 Pydantic 在图状态中的本质差异。

### 代码中的实际方案
`debate/state.py`：
```python
class DebateState(TypedDict, total=False):
    # ... 所有字段都是 Optional 的，因为 total=False
    topic: str
    current_round_arguments: Annotated[list[AgentArgument], add]
    final_report: Optional[DebateReport]
```

`AgentArgument`、`DebateReport` 等用 Pydantic `BaseModel`：
```python
class AgentArgument(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)  # Pydantic 验证
    key_claims: list[str] = Field(min_length=3)  # Pydantic 验证
```

### 如何对面试官表述
> "LangGraph 官方推荐用 TypedDict 作为图的状态，原因是 LangGraph 的状态合并机制（reducers）和 Pydantic 的验证机制会冲突。Pydantic BaseModel 在每次字段更新时都会触发验证，但 LangGraph 的 `add` reducer 是追加操作，中间状态可能不满足最终的完整性约束，会导致意外的 ValidationError。
>
> TypedDict 只做类型提示，不做运行时验证，LangGraph 可以自由地做增量更新。`total=False` 让所有字段都变成可选的，这样 `graph.invoke({'topic': '...', 'max_rounds': 2})` 不需要提供所有字段，LangGraph 会在执行过程中逐步填充。
>
> Pydantic 的验证能力保留给 LLM 的结构化输出层（`AgentArgument`、`DebateReport`），这里才是真正需要防止 LLM 输出不合法数据的地方。两者各司其职。"

### 亮点
- 能说清楚 TypedDict vs Pydantic 在 LangGraph 里的本质区别（运行时验证 vs 类型提示）
- `total=False` 的具体作用说得准确

### 瓶颈
- TypedDict 不做运行时验证，图内部的类型错误只能在运行时发现，不如 Pydantic 早期暴露
- 没有用 `Annotated` 的 Pydantic validator 做跨字段验证

### 突出的能力
**LangGraph 状态设计的技术理解** + **类型系统的实际工程判断**

---

## Q17：`_make_llm()` 每次调用都创建一个新的 `ChatAnthropic` 实例，为什么不做成单例？

*（也可能被问成：LLM 实例是怎么管理的？多个 Agent 并行时会不会有资源竞争？）*

### 面试官想听到的
考察点：**LLM 客户端的生命周期管理**，以及并行场景下的资源安全性。

### 代码中的实际方案
`debate/llm.py`：
```python
def _make_llm() -> ChatAnthropic:
    kwargs = {"model": MODEL_ID}
    if base_url := os.getenv("ANTHROPIC_BASE_URL"):
        kwargs["base_url"] = base_url
    # ... 每次调用都 new 一个 ChatAnthropic
    return ChatAnthropic(**kwargs)
```

`debate/nodes/agents.py`：
```python
def _agent_node(state: dict, role: str) -> dict:
    llm = _make_llm()  # 每个 agent node 调用时都创建新实例
    structured_llm = llm.with_structured_output(AgentArgument, include_raw=True)
```

### 如何对面试官表述
> "这是一个有意识的取舍。`ChatAnthropic` 本质上是一个 HTTP 客户端的薄封装，它本身是无状态的——不持有连接池、不持有会话状态，每次 `invoke` 都是独立的 HTTPS 请求。所以每次 new 一个实例的开销非常小（只是 Python 对象创建），不存在资源浪费。
>
> 做成单例反而有一个潜在风险：`with_structured_output()` 会返回一个新的 Runnable chain，如果单例在多个 Agent 并行调用时被共享，chain 的内部状态（如 retry counter）可能出现竞态。每次创建新实例可以完全避免这个问题。
>
> 真正需要做成单例的是 `sentence-transformers` 的模型（`_get_model()` 里已经是单例了），因为它加载了 130MB 的权重到内存，创建成本高、内存占用大。LLM 客户端没有这个问题。"

### 亮点
- 能区分"创建成本高"（sentence-transformers）和"创建成本低"（ChatAnthropic）的不同处理策略
- 指出单例在并行场景下的潜在竞态风险

### 瓶颈
- 如果 Anthropic SDK 未来引入连接池复用，每次 new 实例会绕过连接复用的优化
- 没有对 LLM 实例做健康检查或熔断

### 突出的能力
**资源生命周期管理的工程判断** + **并发安全的主动思考**

---

## Q18：`RoundRecord` 里的 `divergence_score` 初始值是 0.0，然后被 `divergence_check_node` 回填。这个设计有什么问题？

*（也可能被问成：divergence_score 是什么时候写入的？collect 和 divergence_check 的职责分离是否合理？）*

### 面试官想听到的
考察点：**数据一致性意识**，能否发现"中间状态不一致"的设计问题，并提出改进方案。

### 代码中的实际方案
`debate/nodes/collect.py` 创建 RoundRecord 时 divergence_score=0.0（占位）：
```python
new_record = RoundRecord(
    round_num=round_num,
    arguments=current_args,
    divergence_score=0.0,  # 占位，由 divergence_check_node 回填
)
return {"round_history": round_history + [new_record], ...}
```

`debate/nodes/divergence_check.py` 回填：
```python
score, pairs = compute_divergence(latest_round.arguments)
updated_record = latest_round.model_copy(update={"divergence_score": score})
updated_history = list(round_history[:-1]) + [updated_record]
return {"round_history": updated_history, "divergence_score": score, ...}
```

### 如何对面试官表述
> "这个设计有一个数据一致性窗口：在 `collect_round1` 执行完到 `divergence_check_node` 执行完之间，`round_history` 里最新的 `RoundRecord.divergence_score` 是 0.0，这是一个错误的值。如果在这个窗口内有人读 `round_history`（比如 Streamlit 的 streaming 渲染），会看到 divergence_score=0.0。
>
> 更深的问题是职责分离不彻底：`collect_round1` 创建了一个不完整的 `RoundRecord`，依赖下游节点来补全。这违反了'每个节点的输出应该是完整的'原则。
>
> 更干净的设计是把 divergence 计算移到 `collect_round1` 里，直接在创建 `RoundRecord` 时就算好 `divergence_score`，然后 `divergence_check_node` 只做路由决策，不修改历史数据。这样 `round_history` 里的数据始终是完整的，也避免了 `model_copy` 重建历史列表的开销。"

### 亮点
- 发现了"中间状态不一致窗口"这个微妙问题
- 提出了更干净的重构方向（计算移到 collect，check 只做路由）

### 瓶颈
- 把 divergence 计算移到 collect 会让 collect 节点变重，职责增加
- 现有设计在单线程图里问题不大，因为两个节点是顺序执行的

### 突出的能力
**数据一致性的细节意识** + **职责分离原则的工程判断**

---

## Q19：`route_divergence` 返回类型是 `list[Send] | str`，LangGraph 是怎么区分这两种返回值的？

*（也可能被问成：routing function 能同时返回 Send 列表和字符串吗？LangGraph 内部是怎么处理的？）*

### 面试官想听到的
考察点：**LangGraph 1.x 的 routing function 机制**，以及对框架内部行为的深入理解。

### 代码中的实际方案
`debate/nodes/dispatch.py`：
```python
def route_divergence(state: DebateState) -> list[Send] | str:
    if round_num >= max_rounds:
        return "synthesize_stub"       # 返回字符串 → 路由到指定节点
    if divergence_score < DIVERGE_THRESHOLD:
        return "synthesize_stub"
    return _build_rebuttal_sends(state)  # 返回 list[Send] → 并行 fan-out
```

`debate/graph.py`：
```python
builder.add_conditional_edges(
    "divergence_check_node",
    route_divergence,
    # 没有显式的 path_map，LangGraph 直接接受 routing function 返回值
)
```

### 如何对面试官表述
> "LangGraph 1.x 的 `add_conditional_edges` 对 routing function 的返回值有两种处理逻辑：
>
> 1. 如果返回 `str`，就把它当作目标节点的名字，路由过去
> 2. 如果返回 `list[Send]`，就把每个 `Send` 作为独立的并行任务分发，这是 fan-out 的实现方式
>
> 这两种返回类型是 LangGraph 在同一个 superstep 内的两种调度模式，框架通过 `isinstance(result, list)` 来区分。这个特性是在 LangGraph 1.1.9 里验证过的——我在研究过程中专门去读了 `pregel/main.py` 确认了这个行为。
>
> 需要注意的是，如果 routing function 被注册成了 node（`builder.add_node`），LangGraph 会把返回值当成 state update dict 来处理，`list[Send]` 不是合法的 state update，会报 `InvalidUpdateError`。这就是为什么 routing function 只能传给 `add_conditional_edges`，不能注册成 node。"

### 亮点
- 能说清楚 LangGraph 内部的两种返回值处理机制
- 把之前踩坑的教训（Q2）和这道题联系起来，形成完整认知

### 瓶颈
- 这个行为依赖 LangGraph 内部实现，版本升级可能改变
- 没有官方文档明确说明，需要读源码或踩坑才能知道

### 突出的能力
**对 LangGraph 框架机制的深入理解** + **从踩坑到读源码的学习方式**

---

## Q20：`_build_synthesis_context` 里对每轮的 `reasoning` 字段做了什么处理？为什么不把完整 reasoning 传给 Synthesizer？

*（也可能被问成：Synthesizer 的 prompt 是怎么构建的？context 长度是怎么控制的？）*

### 面试官想听到的
考察点：**LLM prompt 工程的 token 效率设计**，以及信息密度 vs 完整性的取舍。

### 代码中的实际方案
`debate/nodes/synthesize.py` 中 `_build_synthesis_context`：
```python
for round_record in round_history:
    for arg in round_record.arguments:
        if arg.is_sentinel:
            lines.append(f"[{arg.agent_role.upper()}] <no data — sentinel>")
            continue
        # 只传 position + key_claims[:3] + confidence，不传 reasoning
        claims_text = "\n".join(f"    - {c}" for c in arg.key_claims[:3])
        lines.append(
            f"  [{arg.agent_role.upper()}] "
            f"Position: {arg.position} | "
            f"Confidence: {arg.confidence:.0%}\n"
            f"  Key claims:\n{claims_text}"
        )
        # concessions 单独列出
        for c in arg.concessions:
            lines.append(f"    → Conceded to {c.triggered_by_agent}: {c.conceded_point}")
```

### 如何对面试官表述
> "完整的 `reasoning` 字段是每个 Agent 的完整论证文本，通常 200-500 个词，三个 Agent 乘以三轮就是 1800-4500 个词，只是 context 就快撑爆了，还没算 system prompt 和 schema 描述。
>
> Synthesizer 真正需要的是：'各方的核心立场是什么、关键论点是什么、谁让步了什么'。这些信息在 `position`、`key_claims[:3]` 和 `concessions` 里都有。`reasoning` 是支撑论点的展开说明，对 Synthesizer 来说信息密度低、噪声高。
>
> 这是一个有意识的 trade-off：牺牲完整性换取 token 效率和 Synthesizer 的聚焦度。如果 Synthesizer 看到太多冗余信息，反而容易在细节里迷失，无法提炼出真正的共识和分歧。
>
> 当然这也有代价：如果某个关键论据在 `reasoning` 里而不在 `key_claims` 里，Synthesizer 就看不到它。这是 Agent 的 `key_claims` 提取质量对最终报告质量的传导路径。"

### 亮点
- 能量化 token 的问题（1800-4500 词），而不是模糊说"太长了"
- 说清楚 trade-off 的两面：效率 vs 完整性

### 瓶颈
- `key_claims[:3]` 截断可能丢失重要论据（Agent 生成了 7 个 claims 但只传了 3 个）
- Synthesizer 的质量上限被 Agent 的 key_claims 提取质量限制了

### 突出的能力
**LLM prompt 的 token 效率设计** + **信息密度 vs 完整性的 trade-off 意识**

---

## Q21：`convergence_status` 有三个值：`"converged"`、`"max_rounds"`、`"partial"`，`"partial"` 什么时候会出现？

*（也可能被问成：DebateReport 里的 convergence_status 是怎么决定的？有没有可能出现意外的状态？）*

### 面试官想听到的
考察点：**状态机的完备性分析**，能否识别出"防御性 fallback 状态"的存在及其含义。

### 代码中的实际方案
`debate/nodes/synthesize.py` 中 `_determine_convergence_status`：
```python
def _determine_convergence_status(
    divergence_score: float, round_num: int, max_rounds: int
) -> str:
    if round_num < max_rounds and divergence_score < DIVERGE_THRESHOLD:
        return "converged"       # 提前收敛
    elif round_num >= max_rounds:
        return "max_rounds"      # 达到最大轮次
    else:
        return "partial"         # 防御性 fallback，理论上不应出现
```

`debate/state.py`：
```python
convergence_status: Literal["converged", "max_rounds", "partial"]
```

### 如何对面试官表述
> "`partial` 是一个防御性 fallback，在正常流程里理论上不会出现。来分析一下：`synthesize_stub` 只会在 `route_divergence` 返回 `'synthesize_stub'` 时被触发，而 `route_divergence` 只在两种情况下返回字符串：`round_num >= max_rounds`（对应 `max_rounds`）或者 `divergence_score < DIVERGE_THRESHOLD`（对应 `converged`）。
>
> 但 `_determine_convergence_status` 是在 synthesize 节点里独立重新判断的，它不知道 routing 时的具体条件，只能根据当时的 state 字段推断。如果 state 里的 `divergence_score` 和 `round_num` 出现了 routing 时没有的组合——比如 `round_num < max_rounds` 且 `divergence_score >= DIVERGE_THRESHOLD`（意味着'还没到最大轮次但分歧还很高'）——就会落到 `partial`。
>
> 这种情况在正常图执行中不会发生，但如果有人直接构造 state 调用 `synthesize_stub`，或者未来图的路由逻辑改变了，`partial` 就是一个安全网。它的存在是好的工程实践：宁可有一个意义明确的 fallback 状态，也不要让 None 或意外字符串流入报告。"

### 亮点
- 能完整追踪 `convergence_status` 的决定路径，而不是只看代码注释
- 理解"防御性 fallback"的工程价值

### 瓶颈
- `_determine_convergence_status` 和 `route_divergence` 的判断逻辑是重复的，存在不一致的风险——如果 routing 逻辑改了但 synthesis 里的判断没跟着改，会出现 status 不准确

### 突出的能力
**状态机完备性分析** + **防御性编程意识**

---

## Q22：`save_debate` 用了 `INSERT OR REPLACE`，这意味着什么？什么场景下会触发 REPLACE？

*（也可能被问成：如果同一个 debate_id 的辩论跑了两次，数据库里会有几条记录？）*

### 面试官想听到的
考察点：**SQLite upsert 语义的理解**，以及幂等性设计的应用场景。

### 代码中的实际方案
`debate/store.py`：
```python
def save_debate(report: DebateReport, conn=None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO debates "
        "(debate_id, topic, created_at, status, report_json) "
        "VALUES (?,?,?,?,?)",
        (report.debate_id, report.topic,
         report.created_at.isoformat(),
         report.convergence_status,
         report.model_dump_json())
    )
    conn.commit()
```

`debate_id` 是通过 `str(uuid.uuid4())` 生成的（`initialize_node` 里），每次新的 `graph.invoke` 都会生成新 UUID。

### 如何对面试官表述
> "`INSERT OR REPLACE` 是 SQLite 的 upsert 语义：如果 `debate_id`（PRIMARY KEY）已存在，就删除旧行再插入新行；如果不存在，就直接插入。
>
> 在正常使用场景里，`debate_id` 是每次 `graph.invoke` 时由 `initialize_node` 生成的新 UUID，所以每次辩论都会产生唯一的 `debate_id`，REPLACE 分支永远不会触发，等价于普通的 INSERT。
>
> 但 REPLACE 有实际意义的场景是：如果有人手动指定了 `debate_id`（比如测试时），或者未来支持'重跑同一场辩论'的功能，`INSERT OR REPLACE` 保证了幂等性——重跑同一个 `debate_id` 不会报主键冲突，而是更新为最新结果。这是一个'现在不需要但未来可能需要'的防御性设计。
>
> 需要注意的是，SQLite 的 `REPLACE` 实际上是 DELETE + INSERT，不是 UPDATE，所以 `rowid` 会变化。如果有其他表通过 rowid 关联这张表，REPLACE 会破坏关联。当前设计里没有这种关联，所以是安全的。"

### 亮点
- 能说清楚 `INSERT OR REPLACE` 的底层实现（DELETE + INSERT）和 UPDATE 的区别
- 分析了 REPLACE 实际触发的场景和防御性价值

### 瓶颈
- `INSERT OR REPLACE` 的 DELETE + INSERT 语义意味着 `created_at` 也会被更新为新值，如果想保留原始创建时间需要额外处理

### 突出的能力
**SQLite 操作语义的深入理解** + **幂等性设计的工程意识**

---

## Q23：Agent 的 `agent_role` 和 `round_num` 在 `_invoke_with_retry` 里被强制覆写了，为什么？

*（也可能被问成：LLM 输出的结构化字段你会信任吗？哪些字段需要强制校正？）*

### 面试官想听到的
考察点：**对 LLM 结构化输出可靠性的批判性认知**，以及哪些字段必须由代码控制、哪些可以信任 LLM。

### 代码中的实际方案
`debate/nodes/agents.py`：
```python
def _invoke_with_retry(llm, messages, role, round_num, max_retries=2):
    for attempt in range(max_retries + 1):
        result = structured_llm.invoke(messages)
        if result.get("parsed") is not None:
            parsed = result["parsed"]
            # LLM 可能幻想出错误的 role 或 round_num，强制覆写
            parsed.agent_role = role
            parsed.round_num = round_num
            return parsed
```

### 如何对面试官表述
> "这是一个'信任但验证'原则的具体体现。`agent_role` 和 `round_num` 是系统级的元数据，由调用方（`_agent_node`）传入，语义上属于'我知道这个 Agent 是谁、现在是第几轮'，不应该由 LLM 决定。
>
> 但 LLM 在结构化输出里有时会幻觉出错误的值——比如 Pessimist Agent 在 rebuttal 轮次里，LLM 可能在 `agent_role` 字段写 `'optimist'`（因为它在 context 里看到了 Optimist 的论点），或者 `round_num` 写成 0（因为 prompt 里有 Round 1 的历史）。如果不覆写，下游的分析和报告就会出现角色混乱。
>
> 更广泛地说，对于 LLM 的结构化输出，我的原则是：**语义内容（position、reasoning、key_claims）信任 LLM**，因为这是它的专长；**系统元数据（ID、角色、轮次、时间戳）由代码控制**，因为这些值的正确性不依赖语义理解，而依赖调用上下文。"

### 亮点
- 能说出一个具体的 LLM 幻觉场景（Pessimist 写成 optimist）
- 提炼出"语义内容信任 LLM，系统元数据由代码控制"的原则

### 瓶颈
- 强制覆写意味着 LLM 的自我报告被忽略，如果未来需要让 LLM 决定自己的角色（比如动态 persona），这个设计需要调整

### 突出的能力
**对 LLM 结构化输出可靠性的批判性认知** + **系统元数据 vs 语义内容的职责划分**

---

## Q24：`compute_divergence` 里用了 `combinations(arguments, 2)` 做两两比较，如果 Agent 数量从 3 个增加到 5 个，复杂度怎么变化？

*（也可能被问成：分歧检测的时间复杂度是多少？Agent 数量增加对性能有什么影响？）*

### 面试官想听到的
考察点：**算法复杂度分析**，以及对系统扩展性的量化思考。

### 代码中的实际方案
`debate/divergence.py`：
```python
for arg_a, arg_b in combinations(arguments, 2):
    all_claims = arg_a.key_claims + arg_b.key_claims
    embeddings = model.encode(all_claims, normalize_embeddings=True)
    # 每对 agents 做一次 model.encode
    sim_matrix = emb_a @ emb_b.T
    max_sim = float(sim_matrix.max())
```

### 如何对面试官表述
> "当前是 3 个 Agent，`combinations(3, 2) = 3` 对，每对做一次 `model.encode`，总共 3 次 embedding 计算。
>
> 如果增加到 5 个 Agent，`combinations(5, 2) = 10` 对，embedding 计算次数变成 10 次，增长是 O(n²)。但更大的问题是 `model.encode` 的调用次数：当前每对 agents 单独调用一次，如果 3 个 Agent 各有 5 个 claims，每次 encode 处理 10 个 tokens，10 对就是 100 次 token 编码。
>
> 优化方向是把所有 Agent 的 claims 合并成一个 batch，做一次 `model.encode`，然后切片做矩阵乘法：
> ```python
> all_claims = [c for arg in arguments for c in arg.key_claims]
> embeddings = model.encode(all_claims, normalize_embeddings=True)
> # 切片拿到每个 agent 的 embedding，再做 pairwise 矩阵乘法
> ```
> 这样 embedding 计算从 O(n²) 降到 O(n)，矩阵乘法的 O(n²) 不变，但 GPU/BLAS 做矩阵乘法比多次调用 encode 快得多。
>
> 对于 3 个 Agent 的当前规模，差异可以忽略；如果扩展到 10+ 个 Agent，这个优化就值得做了。"

### 亮点
- 能量化复杂度（3对 vs 10对），而不是模糊说"变慢了"
- 提出了具体的 batch encoding 优化方案，有代码

### 瓶颈
- batch encoding 的优化需要重构 `compute_divergence` 的内部实现，增加代码复杂度
- 3 个 Agent 的场景下，优化收益为零

### 突出的能力
**算法复杂度分析** + **性能优化的工程判断（何时优化，何时不优化）**

---

## Q25：`graph.stream(stream_mode="updates")` 和 `stream_mode="values"` 有什么区别？你为什么选 `"updates"`？

*（也可能被问成：LangGraph 的 stream_mode 有哪些选项？各自返回什么？）*

### 面试官想听到的
考察点：**LangGraph streaming API 的深入理解**，以及 UI 渲染场景下的选型判断。

### 代码中的实际方案
`app.py`：
```python
for chunk in graph.stream(
    {"topic": topic, "max_rounds": max_rounds},
    config={...},
    stream_mode="updates"   # 每个 node 完成时推送 {node_name: state_delta}
):
    for node_name, node_update in chunk.items():
        _render_agent_chunk(node_name, node_update)
```

### 如何对面试官表述
> "LangGraph 的 stream_mode 主要有三种：
>
> - `'values'`：每个 node 完成后推送**完整的当前 state**，包括所有字段
> - `'updates'`：每个 node 完成后只推送**这个 node 写入的 delta**，格式是 `{node_name: {changed_fields}}`
> - `'debug'`：推送所有内部事件，包括 checkpoint、task 等，用于调试
>
> 我选 `'updates'` 的原因有两个：
>
> 第一，**UI 渲染只需要知道'谁刚完成了、写了什么'**，不需要每次都拿完整 state。`'values'` 模式每次推送的是全量 state，包括所有 Agent 的历史 arguments、round_history 等，数据量很大，而且大部分是上一个 chunk 里已经有的旧数据。
>
> 第二，**按 node_name 分发渲染逻辑更自然**。`'updates'` 的 chunk 格式 `{node_name: delta}` 天然支持 switch-case 风格的渲染分发：agent node 来了渲染 Agent 卡片，divergence_check 来了显示分歧分数。用 `'values'` 的话，每次都要 diff 前后两个完整 state 才能知道哪个 Agent 刚完成，逻辑更复杂。"

### 亮点
- 能说清楚三种 stream_mode 的区别，不只是知道用 `"updates"`
- 选型理由有两个维度：数据量 + 渲染逻辑

### 瓶颈
- `"updates"` 模式下，如果一个 node 没有写任何 state 字段（比如 `save_node` 返回 `{}`），chunk 里会有一个空 dict，UI 渲染要处理这个 edge case

### 突出的能力
**LangGraph streaming API 的深入理解** + **UI 渲染场景下的技术选型判断**

---

## Q26：如果要给这个系统加"人工介入"（Human-in-the-Loop）功能，比如让用户在某轮后决定是否继续辩论，你会怎么实现？

*（也可能被问成：LangGraph 的 interrupt 机制是什么？怎么在图执行中途暂停等待用户输入？）*

### 面试官想听到的
考察点：**LangGraph Human-in-the-Loop 的实现机制**，以及对 checkpointer 的理解。

### 代码中的实际方案
`debate/graph.py` 中已经有 checkpointer：
```python
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

但目前没有使用 `interrupt()` 函数。

### 如何对面试官表述
> "LangGraph 的 Human-in-the-Loop 核心是 `interrupt()` 函数和 checkpointer 的配合。`interrupt()` 在节点执行中途抛出一个特殊的信号，LangGraph 会把当前 state 保存到 checkpointer，然后暂停执行，把控制权还给调用方。
>
> 具体实现：在 `divergence_check_node` 里加一个 interrupt 点：
> ```python
> from langgraph.types import interrupt
>
> def divergence_check_node(state: DebateState) -> dict:
>     score, pairs = compute_divergence(latest_round.arguments)
>     # 每轮结束后询问用户是否继续
>     user_decision = interrupt({
>         'divergence_score': score,
>         'round_num': state['round_num'],
>         'message': f'Round {state["round_num"]} complete. Divergence: {score:.2f}. Continue?'
>     })
>     if user_decision == 'stop':
>         # 强制进入 synthesize
>         return {..., 'max_rounds': state['round_num']}
>     return {...}
> ```
>
> 调用方（Streamlit）用 `graph.stream` 执行到 interrupt 时会收到 `GraphInterrupt` 事件，展示给用户，然后用 `graph.invoke(Command(resume=user_decision), config=thread_config)` 从断点继续执行。
>
> 关键前提是 `checkpointer` 必须存在，否则 `interrupt()` 会报错。代码里已经有 `InMemorySaver`，所以可以直接加 interrupt，不需要改 graph 的编译方式。"

### 亮点
- 能写出完整的 `interrupt()` 代码，说明真正理解了机制
- 指出 checkpointer 是前提条件，代码里已经满足

### 瓶颈
- `InMemorySaver` 在进程重启后丢失 state，如果用户关闭浏览器再回来，无法恢复 interrupt 状态
- 生产环境需要换成 `SqliteSaver` 或 `PostgresSaver` 才能支持跨进程的 interrupt 恢复

### 突出的能力
**LangGraph Human-in-the-Loop 的完整实现能力** + **checkpointer 与 interrupt 关系的深入理解**

---

*文档维护：有新问题随时补充在对应分类下。*
