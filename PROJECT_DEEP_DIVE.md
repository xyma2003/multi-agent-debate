# Multi-Agent Debate System — 深度解析

## 一、项目背景：它解决什么问题？

单个 LLM 做多角度分析时，本质上是**假装多角色**——同一个模型在 Round 1 说"这件事有风险"，Round 2 又说"但是机会也很大"，最终两边和稀泥。这叫 **sycophancy（讨好行为）**：模型倾向于生成"平衡的"、让人舒服的答案，而不是真正有分歧的立场。

这个项目的核心思路是：**强制隔离 → 真实对立 → 语义检测分歧是否消除 → 可审计报告**。

---

## 二、整体架构

```
graph.py (StateGraph)
│
├── state.py        ← 所有数据结构的定义（单一来源）
├── prompts.py      ← 三个 Agent 的 system prompt
├── llm.py          ← Claude 实例工厂
├── divergence.py   ← 语义分歧计算（sentence-transformers）
│
└── nodes/
    ├── initialize.py      ← 初始化 debate_id、round_num 等
    ├── dispatch.py        ← 路由函数（不是节点，是条件边）
    ├── agents.py          ← 三个 Agent 节点的共享实现
    ├── collect.py         ← Fan-in 汇集节点
    ├── divergence_check.py← 执行分歧检测，写入 state
    ├── synthesize.py      ← 最终报告生成
    └── save.py            ← 持久化到 SQLite
```

整个流程是一个**带循环的 StateGraph**，核心状态在 `DebateState` 里流转。

---

## 三、数据结构（state.py）

理解数据结构是理解整个系统的前提。

### 3.1 核心状态：DebateState（TypedDict）

```
topic          ← 输入话题（外部传入）
max_rounds     ← 最大轮次（外部传入）
debate_id      ← initialize 节点生成
round_num      ← 当前轮次（0-indexed），每轮 +1
current_round_arguments  ← 【特殊】带 add reducer 的累加器
round_history  ← 每轮完成后的 RoundRecord 列表
divergence_score        ← 最新轮的分歧分
diverged_pairs          ← 哪两个 agent 之间分歧
final_report   ← 最终 DebateReport
status         ← "running" / "converged" / "max_rounds"
```

**关键设计**：`current_round_arguments` 用 `Annotated[list[AgentArgument], add]`，这是 LangGraph 的 reducer 机制。三个 Agent 并行跑，每个都往这个字段 append 自己的结果，LangGraph 自动合并，不会互相覆盖。其他字段都是 last-write-wins。

### 3.2 AgentArgument（Pydantic）

每个 Agent 每轮输出一个结构化的 `AgentArgument`：

```
agent_role     ← "optimist" / "pessimist" / "devil"
round_num      ← 本轮轮次
position       ← 一句话核心立场（禁止 hedge）
reasoning      ← 完整论证
confidence     ← 自报置信度 0.0-1.0
key_claims     ← 3-7 条具体论点（用于 embedding）
concessions    ← 本轮让步列表
is_sentinel    ← True = LLM 解析失败后注入的兜底对象
```

`key_claims` 是整个分歧检测的基础，专门做 embedding 用。`concessions` 是让步记录，每条都有精确的归因（是谁的哪条论点让你让步的）。

### 3.3 Concession（Pydantic）

```
conceded_point      ← 你让步了什么
triggered_by_agent  ← 是哪个 agent
triggered_by_claim  ← 是对方哪条具体 claim 让你让步的
rationale           ← 一句话说明为什么让步
```

这是推理链可审计的核心——每一次立场改变都有完整的因果记录。

---

## 四、反 sycophancy 设计（prompts.py）

这是整个项目最核心的工程决策。

### 4.1 方法论驱动，而非强度驱动

**错误做法**：`"You are very pessimistic."`  
**正确做法**：`"Your analytical framework is: 1. Identify the single most likely failure mode..."`

区别在于：强度驱动让 LLM 扮演一个性格角色，很容易在压力下 collapse（"好吧你说的有道理，我同意"）。方法论驱动是给 LLM 一套**分析工具**，它是在执行程序，不是在表演性格。

### 4.2 PROHIBITION 硬约束

三个 Agent 的 prompt 里都有明确的 `PROHIBITION` 段：

- **Optimist**：禁止提风险、caveats、failure modes。禁止写 "however", "but", "although", "unless", "on the other hand" 等词。
- **Pessimist**：禁止提 upsides、机会、增长潜力。
- **Devil**：禁止同意主流观点，哪怕部分同意。

这些不是软性引导，是硬性禁令——直接命名哪些词不能出现。

### 4.3 Round 1 并行扇出保证认知隔离

三个 Agent 在第一轮是**完全并行**的（LangGraph `Send` fan-out），相互看不到对方的输出。这保证了 Round 1 的立场是独立形成的，不存在锚定效应。

后续 rebuttal 轮才会把对手的 `key_claims` 和 `position` 注入 human message，但仍然带着让步指令："只有对方论点**逻辑上优于**你的立场时才让步，不能为了表现平衡而让步。"

---

## 五、辩论流程（graph.py）

```
START
  │
  ▼
initialize_node        ← 生成 debate_id，设 round_num=0
  │
  ▼  (conditional edges via dispatch_round1)
  ├─── Send → optimist_node  ─┐
  ├─── Send → pessimist_node ─┤  (并行)
  └─── Send → devil_node    ─┘
                              │
                              ▼
                        collect_round1   ← fan-in，打包成 RoundRecord，round_num+1
                              │
                              ▼
                    divergence_check_node ← 计算分歧分，写 state
                              │
                   [route_divergence 路由函数]
                        /             \
               diverged              converged 或 max_rounds
                  │                        │
         Send fan-out (rebuttal)      synthesize_stub
         → 三 agent rebuttal              │
         → collect_round1 (复用)       save_node
         → divergence_check_node          │
         → (循环)                        END
```

**关键细节**：

- `dispatch_round1` 和 `route_divergence` **不是节点**，是路由函数，传给 `add_conditional_edges`。这是因为它们返回 `list[Send]`（并行任务列表），如果注册成节点会报 `InvalidUpdateError`。
- `collect_round1` 被复用——Round 1 和所有 rebuttal 轮共用同一个 fan-in 节点。
- 整个图用 `InMemorySaver` 做 checkpoint，支持 LangGraph 的 `interrupt()` 能力（人工介入）。

---

## 六、分歧检测（divergence.py）

用 `sentence-transformers` 的 `BAAI/bge-small-en-v1.5` 模型（~130MB，本地运行）。

### 为什么不用 Claude API 做比较？

每次分歧检测都在辩论循环的内层，如果用 API 调用，每轮多 2 个 round-trip，延迟和成本都不可接受。本地 embedding 一次批量 encode 所有 claims，几毫秒完成。

### 为什么在 key_claims 上做，而不是全文？

全文 embedding 会按话题聚类——三个 Agent 都在讨论同一个 topic，全文语义必然相似，看不出分歧。`key_claims` 是提炼出来的具体论点，语义差距更显著。

### 检测逻辑

```python
for arg_a, arg_b in combinations(arguments, 2):
    # 跨 agent 的 claims 做 embedding
    # 取 claim-level 最大余弦相似度
    max_sim = sim_matrix.max()

    if max_sim > 0.97:  # 快速收敛路径
        # 不记录为 diverged
    elif max_sim < 0.75 or max_sim <= 0.97:
        # diverged_pairs 记录这对 agent

divergence_score = 1.0 - mean(pairwise_max_sims)
```

分歧分 `0.0` = 完全收敛，`1.0` = 完全分歧。阈值 0.75 以上（非快速路径）都被保守地处理为分歧。

---

## 七、终止条件与置信度公式（synthesize.py）

### 7.1 终止不靠轮次计数

`route_divergence` 里的判断：
- `divergence_score < 0.75`（DIVERGE_THRESHOLD）→ 收敛，终止
- `round_num >= max_rounds` → 达到上限，强制终止

**不是"跑满 N 轮就结束"**，而是真正检测到观点趋同后才终止。

### 7.2 置信度公式（纯 Python，SYNTH-03）

```python
max_divergence = max(r.divergence_score for r in round_history)
round_adjustment = {1: 1.0, 2: 0.9, 3: 0.8}.get(round_num, 0.8)
confidence_score = round((1.0 - max_divergence) * round_adjustment, 4)
```

两层含义：
- `(1 - max_divergence)`：整个辩论过程中最大分歧越大，置信度越低——说明各方分歧从未真正消除
- `* round_adjustment`：需要越多轮才能收敛，置信度惩罚越重

**最关键的约束**：`confidence_score` **不在 LLM 的输入 prompt 里，也不在 `SynthesizerOutput` 的 schema 里**。`SynthesizerOutput` 只有 `consensus_points`、`disputed_points`、`verdict`。置信度是 LLM 调用完成后，由 Python 代码算出来的。这叫 SYNTH-03 不变式，保证数字可复现、可审计、不受 LLM 幻觉污染。

---

## 八、最终报告（DebateReport）

```
debate_id, topic, created_at
consensus_points      ← 三方真正达成共识的点
disputed_points       ← 仍有分歧的点（含每个 agent 的具体立场）
verdict               ← 2-4 句最终陈述
confidence_score      ← Python 公式算出的 0-1 分
convergence_status    ← "converged" / "max_rounds" / "partial"
reasoning_trace       ← 完整 RoundRecord 列表（每轮每人的完整论证）
concession_log        ← 所有轮次让步记录的扁平化列表
```

`reasoning_trace` 是"可审计"的核心——你可以追溯整个推理过程，看任何一轮任何一个 Agent 说了什么，以及谁因为什么让步了。

---

## 九、工程亮点总结

| 问题 | 解决方案 |
|------|---------|
| 单 LLM 多角色趋同 | 方法论 prompt + PROHIBITION 块 + Round 1 完全并行隔离 |
| 分歧检测不可靠 | 在 key_claims（而非全文）上做 embedding，本地模型，无 API 延迟 |
| 置信度被 LLM 乱填 | SYNTH-03：公式硬编码在 Python，不进 LLM prompt |
| 让步来源不可追溯 | Concession 模型强制记录 `triggered_by_agent` + `triggered_by_claim` |
| 并行 fan-out state 竞争 | `Annotated[list, add]` reducer，只有 `current_round_arguments` 用，其余 last-write-wins |
| LLM 输出解析失败崩溃 | `include_raw=True` + 3 次重试 + sentinel 对象注入，图永不崩溃 |
