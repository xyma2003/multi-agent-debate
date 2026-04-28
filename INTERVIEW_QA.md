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

*文档维护：有新问题随时补充在对应分类下。*
