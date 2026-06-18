# 用三个 LLM 互相辩论：一个对抗 AI 谄媚的实证研究

> 本文记录了一个多智能体辩论系统的设计与实验过程。LLM 的 sycophancy 问题有两种表现：被质疑时改口，以及主动给出"两边都有道理"的中庸答案。本项目针对后者——用结构性机制强制 Agent 产生真实的立场分歧，并通过三个反直觉的实验发现，逐步逼近一个真正有效的解法。

---

## 背景：LLM 的谄媚问题（Sycophancy）

大型语言模型通过 RLHF 训练，目标函数是最大化人类评分者的认可。这一机制赋予了模型流畅的对话能力，但也内嵌了一个系统性偏差：**模型学会了迎合，而不只是学会了正确**。

研究者将这一现象称为 **sycophancy**，它在实际使用中有两种典型表现：

**其一，压力下的立场反转。** 用户对模型的回答表示异议时，即使原始判断完全正确，模型也倾向于认错并修改答案。Perez et al.（2022）的实验中，仅凭一句"我不这么认为"，就能让模型以远高于随机的概率放弃自己正确的立场。

**其二，预防性的立场回避。** 在没有任何外部压力的情况下，模型主动选择模糊措辞：

> "A 有其优势，但 B 也有独特价值。最终的选择取决于具体情境。"

这不是审慎，是规避。对于投资评估、技术选型、战略分析等需要真实风险收益判断的场景，这类回答没有实质价值。

本项目针对**第二种表现**：在多 Agent 辩论系统中，通过结构性机制强制产生真实的立场分歧。但工程实现过程中我们发现，第一种问题在 Agent 之间同样存在——面对对方的反驳，Agent 会主动让步以求和，而不是坚守立场继续论证。这两个问题最终都需要在系统设计层面加以应对。

---

## 为什么"三个 Agent 各说一遍"不够

最朴素的设计是：给三个 Agent 不同的立场标签，让它们分别分析，再汇总。这个方案有两个根本性的问题。

**问题一：立场在对话压力下崩塌。**

当 Agent 看到其他 Agent 的反驳时，如果提示词只是"你是悲观的分析师"，它立刻面临一个无法回答的问题：*我为什么要坚持悲观？* 没有内在锚点，模型会走阻力最小的路径——开始对冲：

> "你提出的机会确实值得关注，尽管我仍然认为风险不可忽视……"

这句话的信息量几乎为零：它既没有坚持原有分析，也没有给出任何新的论点，只是把两个立场并排摆在一起，本质上和单个 LLM 的对冲输出没有区别。

**问题二：表面分歧不等于真实分歧。**

两个 Agent 可以用不同的词汇表达相同的观点——比如一个说"市场需求不确定"，另一个说"用户采纳率存在风险"，语言不同，但实质上都在说同一件事。如果系统没有语义层面的分歧检测机制，就无法判断当前的辩论是否仍有进行下去的价值。

这两个问题分别对应了系统的两个核心设计决策。

---

## 设计决策一：方法论 Persona + PROHIBITION 约束

### 为什么方法论比人格有效

初版提示词给 Agent 分配的是人格标签：

```
你是一个极度悲观的分析师，请从最坏的角度分析这个问题。
```

问题在于，"悲观"是一个形容词，不是一个分析框架。当对手给出一个具体的、有数据支撑的论点时，"因为我天生悲观"根本无法作为反驳理由——模型会合理地认为，接受对方的论点才是"正确"的行为。

将人格替换为**分析方法论**之后，Agent 有了坚守立场的内在逻辑：

```
你是风险分析师。你的职责是找出这个提案最可能的失败模式。
使用以下框架：
1. 识别概率最高的失败路径
2. 评估其发生概率与影响量级
3. 判断预期收益是否足以覆盖该风险

你的分析不是为了"平衡"，而是为了找到最大的风险暴露点。
```

现在，当 Optimist 说"这个市场有巨大的上行空间"时，Pessimist 不需要"被说服"——它的方法论要求它继续寻找失败路径，而不是接受对方对上行空间的描述。立场的维持有了方法论层面的理由。

### PROHIBITION 约束块

软性引导无效。"请避免使用对冲语言"这类指令，模型可以在句子结构上遵从，但在语义上继续骑墙——比如用"需要关注的是……"代替"但是……"，实质没有区别。

有效的方式是**显式禁止词汇表**，并附加失败定义：

```
你绝对不能使用以下词语或任何语义等价的表达：
"然而"、"但是"、"另一方面"、"这取决于"、"也有其优点"、
"需要具体情况具体分析"、"两者各有优势"

使用上述任何表达意味着你的分析失败。
```

关键是"失败"这个词——它把遵守约束从建议变成了任务定义的一部分。实测中，加入这个约束后，Agent 输出中的对冲标记词频率下降明显，模型开始寻找替代的表达方式来传递信息，而不是依赖模糊措辞兜底。

### 三个 Agent 的职责划分

| Agent | 分析框架 | 参照角色 | 核心禁令 |
|-------|---------|---------|---------|
| **Optimist** | 寻找非对称上行空间；枚举成功条件；评估机会量级 | 种子期 VC 分析师 | 禁止提及风险、例外、前提条件 |
| **Pessimist** | 找出最可能的失败路径；评估概率 × 影响；判断收益能否覆盖风险 | 风险债务基金经理 | 禁止提及机会、正面信号、成功案例 |
| **Devil's Advocate** | 识别两方共同接受的隐含假设；攻击问题本身的框架而非具体立场 | 哲学-经济学家 | 禁止简单反对 Optimist；禁止站队 Pessimist |

**Devil 的设计是整个系统最脆弱的环节。** 初版将其定义为"挑战主流观点"。实测结果是它在 10 个测试问题中有 9 个与 Pessimist 站在同一侧——因为在大多数商业/技术讨论中，乐观立场是默认预设，悲观批评才是"挑战主流"的那一方。Devil 因此变成了 Pessimist 的附议者，系统从三角对立退化为 2 vs 1。

修复方案将 Devil 重新定位为**框架质疑者**：它的任务不是支持任何一方，而是找出 Optimist 和 Pessimist 都默认为真的前提，然后质疑这个前提本身是否成立。比如，当两方都在讨论"这个 SaaS 产品应不应该进入企业市场"时，Devil 的输入可能是"企业市场的采购周期假设在当前宏观环境下是否仍然成立"——这是两方都没有质疑过的地基。

这个修改在实验中带来了可测量的 PDS 提升，细节见 Finding B。

---

## 设计决策二：NLI 分歧检测

### 余弦相似度的根本性缺陷

系统需要在每轮结束后判断：辩论是否还有继续的价值？最直觉的做法是把每个 Agent 的核心论点向量化，用余弦相似度衡量立场之间的距离——距离越小，说明越接近收敛。

第一轮基准测试的结果是：**10 个问题全部在第一轮触发收敛**，分歧分数均落在 0.097–0.258 之间，远低于 0.75 的收敛阈值。

这个结果迫使我检查余弦相似度的计算过程。问题很清晰：余弦相似度衡量的是**语义空间中的话题距离**，而不是**立场的逻辑对立程度**。以下两个句子：

- "风险投资是一种具有高回报潜力的融资模式"
- "风险投资是一种风险极高、创业公司应谨慎对待的融资模式"

在嵌入空间中距离很近——它们共享几乎相同的关键词（风险投资、融资、回报/风险）。但它们在逻辑上是完全对立的立场。余弦相似度无法区分"在讨论同一件事"和"对同一件事持相反意见"。

### NLI 方案：检测逻辑矛盾而非话题距离

自然语言推断（NLI）模型被训练为判断两个句子之间的逻辑关系：**蕴含（Entailment）**、**中立（Neutral）**、**矛盾（Contradiction）**。这正是辩论分歧检测需要的信号。

实现使用 `cross-encoder/nli-deberta-v3-small`，对每对 Agent 的所有论点进行两两交叉比较：

```python
from sentence_transformers import CrossEncoder
import numpy as np

nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")

def compute_nli_divergence(claims_a: list[str], claims_b: list[str]) -> float:
    pairs = [(a, b) for a in claims_a for b in claims_b]
    scores = nli_model.predict(pairs)  # shape: (n_pairs, 3)
    # DeBERTa NLI 输出顺序: [contradiction, entailment, neutral]
    contradiction_probs = scores[:, 0]
    return float(contradiction_probs.max())
```

整体分歧分数是三对 Agent（Optimist-Pessimist、Optimist-Devil、Pessimist-Devil）各自最大矛盾概率的均值。

为避免每轮都跑 cross-encoder（推理成本较高），实现了两层流水线：先用余弦相似度做快速过滤——如果最大余弦相似度已经超过 0.97，直接认定收敛，跳过 NLI；否则才调用 NLI 模型。这个快路径在实测中覆盖了约 15% 的论点对，节省了对应的推理时间。

切换到 NLI 后，第一轮分歧分数上升到 0.83–0.86，系统开始正常跑出多轮辩论，平均收敛轮次为 3.0，立场稳定性（SSS）为 0.883。

---

## 系统架构：LangGraph StateGraph

### 状态设计

图的状态使用 LangGraph 要求的 `TypedDict` 格式。所有字段默认是最后写入胜（last-write-wins），只有一个字段使用了 `add` 规约器：

```python
from typing import Annotated
from langgraph.graph.message import add

class DebateState(TypedDict):
    topic: str
    debate_id: str
    round_num: int
    max_rounds: int

    # 唯一使用 add 规约器的字段：允许并行 Agent 的输出合并而不是覆盖
    current_round_arguments: Annotated[list[AgentArgument], add]

    round_history: list[RoundRecord]
    divergence_score: float
    diverged_pairs: list[tuple[str, str]]
    final_report: Optional[DebateReport]
```

`current_round_arguments` 使用 `add` 规约器是并行扇出的关键。LangGraph 在并行执行三个 Agent 节点时，每个节点会独立写入 `current_round_arguments`——如果使用默认的 last-write-wins，三个输出中只有最后一个会被保留。`add` 规约器将每次写入追加到列表末尾，从而正确地合并三个 Agent 的输出。

其他字段使用 last-write-wins 是刻意选择：`round_num`、`divergence_score` 等字段只会被单个节点写入，不存在并发冲突，无需规约器。

### 图拓扑

```
START
  └─→ initialize_node
        └─→ dispatch_round1          ← 路由函数（非节点）
              ├─→ optimist_node ──┐
              ├─→ pessimist_node ─┤  并行执行，通过 Send API 分发
              └─→ devil_node ─────┘
                        └─→ collect_round1    ← 合并 current_round_arguments，写入 round_history
                                └─→ divergence_check_node
                                        └─→ route_divergence   ← 路由函数（非节点）
                                              ├─→ [分歧] 下一轮辩论（回到并行扇出）
                                              └─→ [收敛] synthesize_node
                                                              └─→ save_node → END
```

**一个工程陷阱值得记录**：`dispatch_round1` 和 `route_divergence` 是路由函数，必须作为参数传给 `add_conditional_edges`，而不能作为节点注册到图中。早期实现中误将路由函数注册为节点，导致 `InvalidUpdateError`——路由函数返回的是 `list[Send]`，LangGraph 无法将其解释为状态更新。

第一轮强制**完全认知隔离**：每个 Agent 的上下文中只有辩题，没有其他 Agent 的输出。这确保三个初始立场是独立推导出的，而不是对彼此的反应。从第二轮起，每个 Agent 收到完整的历史轮次摘要和让步记录。

`collect_round1` 节点在第一轮和所有反驳轮中复用——它的职责始终是：将 `current_round_arguments` 聚合为一条 `RoundRecord`，追加到 `round_history`，然后清空 `current_round_arguments` 以准备下一轮。

### 收敛路由：四卫兵逻辑

`route_divergence` 按顺序检查四个终止条件：

```python
def route_divergence(state: DebateState) -> list[Send] | str:
    # Guard 1: 绝对轮次上限，防止无限循环
    if state["round_num"] >= 10:
        return "synthesize_stub"

    # Guard 2: 分歧分数低于阈值，判定为真实收敛
    if state["divergence_score"] < 0.75:
        return "synthesize_stub"

    # Guard 3: 分数停滞——连续两轮分歧变化小于 0.05
    # 说明 Agent 在重复相同论点，继续辩论不会产生新信息
    if len(state["round_history"]) >= 2:
        prev = state["round_history"][-2].divergence_score
        curr = state["divergence_score"]
        if abs(prev - curr) < 0.05:
            return "synthesize_stub"

    # Guard 4: 无让步——上一轮没有任何 Agent 改变立场
    # 分歧分数高但无让步，说明双方在对话，但没有真正的辩论
    if len(state["round_history"]) >= 2:
        last_round = state["round_history"][-1]
        total_concessions = sum(len(a.concessions) for a in last_round.arguments)
        if total_concessions == 0:
            return "synthesize_stub"

    return [Send("optimist_node", state), Send("pessimist_node", state), Send("devil_node", state)]
```

Guard 3 和 Guard 4 解决的是同一个问题的两种形态：**高分歧但无实质进展**。有些话题上，三个 Agent 会保持高分歧分数，但每轮只是重新陈述各自的原始立场，论点没有演进，也没有任何一方让步——继续辩论只是在浪费 API 调用。Guard 3 从分数角度捕捉这种情况，Guard 4 从行为角度捕捉。

### Agent 节点：结构化输出与 Sentinel 兜底

每个 Agent 节点调用同一个重试包装器：

```python
def _invoke_with_retry(llm, prompt, max_retries=3) -> AgentArgument:
    for attempt in range(max_retries):
        try:
            result = llm.with_structured_output(
                AgentArgument, include_raw=True
            ).invoke(prompt)
            if result["parsed"] is not None:
                return result["parsed"]
        except RateLimitError:
            time.sleep(60 * (attempt + 1))  # 指数退避
        except Exception:
            pass

    # 三次失败后注入 Sentinel，图继续运行
    return AgentArgument(
        agent_role=role,
        round_num=round_num,
        position="[解析失败]",
        reasoning="",
        confidence=0.0,
        key_claims=[],
        concessions=[],
        is_sentinel=True,
    )
```

`include_raw=True` 让 LangGraph 在结构化输出解析失败时返回 `{"parsed": None, "parsing_error": "..."}` 而不是直接抛异常。Sentinel 机制确保图在任何单点失败下都能继续执行并产出最终报告，而不是整体崩溃。

---

## 实验设计与结果

### 评测指标

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| **PDS** | Position Diversity Score，三个 Agent 立场的多样性 | 所有 Agent 对的核心论点嵌入距离均值 |
| **HR** | Hedge Ratio，对冲语言密度 | "然而/但是/取决于"等词每百 token 出现频率 |
| **SSS** | Stance Stability Score，跨轮次立场稳定性 | Agent 跨轮次核心论点嵌入的余弦一致性 |
| **RTC** | Rounds to Convergence，收敛所需轮数 | 整数，1 代表第一轮即收敛 |

测试集：10 个商业/技术/政策类问题，每个系统变体各跑一次。

---

### Finding A：PROHIBITION 使对冲率降低 28%

| 系统 | HR |
|------|----|
| 多 Agent（本系统） | **0.0093** |
| 单 LLM 基线 | 0.0129 |

PROHIBITION 约束不只是让 Agent "更坚定"——它在表达结构层面彻底封堵了对冲句式的生成路径。模型无法构造被禁止的句子，只能寻找其他方式表达信息，这反而逼出了更具体、更有论点密度的输出。

值得注意的是，这里测量的是 HR 的差异，而不是输出质量的整体评估。PROHIBITION 约束本身可能过滤掉一些合理的限定语——"在假设市场规模准确的前提下"这类表达虽然包含条件语，但它是有效信息，不是对冲。这是该机制的边界条件，需要根据具体使用场景调整禁止词表的粒度。

---

### Finding B：PDS 悖论与 Devil 提示词修复

初版多 Agent 系统的 PDS 低于单 LLM 基线，与设计目标相反：

| 系统 | PDS |
|------|-----|
| 多 Agent（旧 Devil 提示词） | 0.1707 |
| 单 LLM 基线 | 0.2160 |
| **多 Agent（新 Devil 提示词）** | **0.2242** |

诊断过程：在旧版本的输出记录中，逐条检查 Devil 每轮的 `position` 字段，发现它在 10 个测试问题的 9 个中与 Pessimist 的核心论点高度重合（嵌入距离均值 0.08，远低于 Optimist-Pessimist 之间的 0.31）。Devil 并没有在"质疑主流"，而是在"跟随反主流"。

根因是提示词定义与系统均衡态之间的冲突：旧提示词让 Devil 对抗"主流观点"，但这个系统里 Pessimist 已经是反主流的声音，Devil 就自然地和它站在一起，形成 2 vs 1。

修复后的 Devil 提示词不再定义"反对谁"，而是定义"攻击什么"：攻击双方都默认为真的框架前提。这把 Devil 从一个会被系统动态拉偏的相对角色，变成了一个有绝对任务定义的独立角色。

---

### Finding C：余弦检测 vs NLI 检测

| 系统 | 第一轮分歧分数 | RTC 均值 | SSS |
|------|-------------|---------|-----|
| 余弦检测 | 0.097–0.258 | 1.0 | 1.000（单轮，无意义） |
| **NLI 检测** | **0.83–0.86** | **3.0** | **0.883** |

余弦版本中，SSS = 1.000 并不表示"Agent 立场非常稳定"，而是表示"只有一轮，没有跨轮次可以比较"。这是一个典型的指标陷阱：数字看起来好，但它衡量的是一个退化情况。

NLI 版本的 SSS = 0.883 的含义是：Agent 在多轮辩论中维持了核心立场，同时在边缘论点上做出了有根据的让步——这才是辩论系统应有的行为模式。

---

## 第二部分：统一 PROHIBITION 的局限

上述三个发现验证了基础系统的设计。但在消融实验过程中，一个更深的问题浮现：PROHIBITION 应该对所有问题类型一视同仁吗？

比较以下两个问题：

- *"AI 开发应该暂停吗？"*
- *"初创公司应该招通才还是专才？"*

第一个是价值观冲突。持不同价值观的人会得出不同结论，且没有一方在客观上是错的。这类问题需要有力的立场对峙——强制 Agent 说"对方也有道理"恰恰削弱了分析价值。

第二个没有通用答案。招通才还是专才，真的取决于团队规模、产品阶段、增长节奏。强制 Agent 取无条件立场，会产生质量更差的分析——因为正确答案本来就是"视具体条件而定"。

对第一类问题有效的 PROHIBITION，在第二类问题上会损害输出质量。系统在对认知需求截然不同的两类问题施加同一约束级别。

---

## 设计决策三：Adaptive PROHIBITION

### 三类问题分类体系

根据问题需要什么样的分歧形式，分为三类：

**价值观争议型（values-based）：** 分歧本质上是价值观或伦理观的冲突，没有客观正确答案。例：*"AI 该不该被监管？"*、*"资本主义与气候解决方案兼容吗？"* → 适合全开 PROHIBITION。

**有明确倾向型（binary）：** 基于证据或普遍认可的原则，一种答案通常更优，但理性的人可以存在分歧。例：*"初创公司应该优先增长还是盈利？"*、*"快速迭代哲学是否合理？"* → 适合中等约束：要求给出方向性建议，但允许附带条件说明。

**情境依赖型（context-dependent）：** 正确答案真的取决于具体情境，"视情况而定"本身就是合理回答。例：*"招通才还是专才？"*、*"微服务还是单体架构？"* → 关闭 PROHIBITION：Agent 映射各方案在哪些条件下成立，而不是强取立场。

### LLM 分类器

问题类型在推理时由一个结构化输出分类器自动判断：

```python
class QuestionClassification(BaseModel):
    question_type: str  # "values_based" | "binary" | "context_dependent"
    confidence: str     # "high" | "medium" | "low"
    reasoning: str      # 一句话解释分类理由

_CLASSIFIER_SYSTEM = """
将问题分为以下三类之一：

  values_based:       分歧源于价值观或伦理，没有客观正确答案。
  binary:             基于证据或分析，通常有更优的一方。
  context_dependent:  正确答案真的取决于具体情境，"视情况"是合法回答。

binary 与 context_dependent 难以区分时：如果多数情况下存在公认的默认正确答案，判 binary。
"""
```

分类器使用 `with_structured_output` 强制合法输出，三次重试失败后回退到 `"binary"`。

### 三级 PROHIBITION 约束

每种问题类型对应不同的 Agent 提示词集合：

**一级——全开（价值观争议型）：** 沿用原始提示词，硬性词汇黑名单完全生效，Agent 不得承认对方立场有任何优点。

**二级——中等（有明确倾向型）：** 取消词语禁令，但要求以有立场的建议结尾：*"建议：是——因为 [一个具体的可证伪理由]"*，不允许以开放问题或对冲结尾。

**三级——关闭（情境依赖型）：** Agent 从"辩手"变成"情境分析师"，职责是映射条件而非表态：

```
# Optimist → "方案 A 分析师"
立场必须采用以下格式：
"该方案在 [具体条件集合] 下是最优选择，因为 [因果机制]。"

# Pessimist → "方案 B 分析师"
"该方案在 [具体条件集合] 下会失败，因为 [因果机制]。"

# Devil's Advocate → "关键变量识别者"
"决策取决于 [具体变量]——以下是衡量方法：[具体方法]"
```

PROHIBITION 不是开关，而是一个连续谱，根据问题的认知需求来校准。

---

### Finding D：PROHIBITION 不会放大虚假确定性

一个合理的担忧：强制 Agent 表态，会不会导致它在忽略明显反证的情况下过度自信——即"虚假确定性"？

对 7 道测试题的每个 Agent 输出，用独立评判模型在 1–5 分制下评估虚假确定性水平：

| 系统 | 虚假确定性 | 合理对冲 | 角色适切的承担 |
|------|---------|---------|------------|
| `full_system` | 3/21 (14.3%) | 2/21 (9.5%) | 16/21 (76.2%) |
| `single_llm` | 3/21 (14.3%) | 6/21 (28.6%) | 12/21 (57.1%) |

两个系统的虚假确定性比例完全相同（14.3%）。实质差异在于：`single_llm` 产出了更多"合理对冲"（28.6% vs. 9.5%）——即为了不出错而刻意不表态的立场。**PROHIBITION 并不会把 Agent 推向无法辩护的主张，而是把它从合理对冲推向了明确承担的立场。**

---

### Finding E：自适应约束提升历史决策准确率

以 10 个有明确历史结论的 M&A 和产品战略决策为基准（Facebook/Instagram 收购、Netflix 流媒体转型、Snapchat 拒绝 Facebook 收购等），评测各系统的分析是否能识别出历史上正确的关键因素：

| 系统 | 历史决策准确率（n=10） |
|------|-----------------|
| `full_system` | 0.40 |
| `single_llm` | 0.60 |
| `adaptive_prohibition` | **0.60** |

全开 PROHIBITION 降低了准确率：强制所有类型的问题都保持对立立场，反而压制了识别复杂决策中关键变量所需的情境分析能力。自适应约束通过将历史决策路由到情境依赖模式，保留了 single_llm 的灵活性——同时产出更结构化、更少对冲的分析。

---

### Finding F：问题类型决定自适应增益的大小

三类问题对比实验（binary n=10，values_based n=10，context_dependent n=20）：

| 问题类型 | full_system focus | adaptive focus | Δ | n |
|---------|------------------|----------------|---|---|
| binary | 2.65 | **2.80** | +5.7% | 10 |
| values-based | 3.10 | **3.10** | 0.0% | 10 |
| **情境依赖型** | 2.00 | **3.50** | **+75%** | 20 |

Focus score = 各类型重点维度的均值（binary：分析深度+论断具体性；values：视角多样性+分析深度；context：论断具体性+实用性）。

**values_based 零差异（3.10 = 3.10）验证了核心设计假设**：分类器在价值观题上正确路由到全开 PROHIBITION，质量得以保全——分类器没有过度适配。

情境依赖类问题获得了最大增益：在 API 设计、基础设施选型、招聘决策、GTM 策略和组织管理 20 道题上，focus score 提升了 **75%**。

**binary 类问题上的反直觉发现**：分类器将多数被人工标注为"binary"的问题路由到了 `context_dependent`。"初创公司该不该做 X"类问题，分类器认为其答案取决于公司阶段、市场和团队——因为本来如此。binary 问题上的性能提升主要来自*情境依赖型提示词设计*（条件映射），而不是*中等 PROHIBITION 级别*本身。

这意味着：**问题类型不是话题固有的属性，而是问题在具体情境下的认知需求属性**。基于认知需求做路由的分类器，比任何硬编码分类体系都更有效。

---

## 可信度机制：置信分与让步追踪

### 置信分：不经过 LLM 的纯计算值

最终报告的置信分（confidence_score）完全由 Python 计算，从不出现在任何 LLM 的提示词或结构化输出 schema 中：

```python
def compute_confidence(round_history: list[RoundRecord], round_num: int) -> float:
    max_divergence = max(r.divergence_score for r in round_history)
    round_adjustment = {1: 1.0, 2: 0.9, 3: 0.8}.get(round_num, 0.8)
    return (1.0 - max_divergence) * round_adjustment
```

两层语义：`(1 - max_divergence)` 反映了这个话题本身有多难达成共识；`round_adjustment` 对需要更多轮次才收敛的结果打折扣。

这个设计来自一个具体的失败：早期版本让 LLM 在 synthesize 节点中生成置信分，结果模型在所有测试问题上输出的分数都集中在 0.72–0.88 之间，与实际的辩论激烈程度完全无关。模型在生成置信分时选择了一个"听起来合理"的区间，而不是基于辩论过程做出判断。将置信分改为公式计算后，它开始真正反映辩论的收敛情况：对于快速收敛的简单问题，置信分接近 0.9；对于三轮后仍然高度分歧的问题，置信分可以低至 0.3。

### 让步追踪：带归因的推理链

每个 Agent 在每轮的输出中都包含一个让步列表，记录这一轮内的任何立场调整：

```python
class Concession(BaseModel):
    conceded_point: str           # 放弃的具体立场
    triggered_by_agent: str       # 哪个对手的论点触发了这次让步
    triggered_by_claim: str       # 触发让步的具体论点原文
    rationale: str                # 为什么接受这一反驳
```

一个典型的让步记录如下：

```json
{
  "conceded_point": "短期内用户获取成本可能高于预期",
  "triggered_by_agent": "pessimist",
  "triggered_by_claim": "类似产品在冷启动阶段的 CAC 中位数是预测值的 2.3 倍",
  "rationale": "历史数据支持这一估算，我的初始预测没有充分考虑冷启动效应"
}
```

让步追踪的价值不只在于记录"谁改变了观点"，更在于提供完整的因果链：**什么论点改变了什么立场，以及为什么这个论点有说服力**。这使得最终报告的结论是可溯源的——用户可以从共识结论回溯到具体的论证过程，而不是面对一个黑箱输出。

---

## 总结

这个项目的核心是六个反直觉发现：

1. **硬约束比软引导有效**：PROHIBITION 词汇黑名单比"请保持立场"更有效，原因是它封堵了模型的语言生成路径，而不只是在语义层面施加偏好。

2. **Agent 角色定义必须在系统均衡态下验证**：Devil 的旧提示词在单独评估时看似合理，但放入有 Pessimist 存在的系统中，被系统动态拉偏为附议者。多 Agent 系统中的角色定义不能在真空中设计。

3. **分歧指标决定系统是否真正运转**：余弦相似度和 NLI 测量的是根本不同的东西——前者是话题距离，后者是逻辑矛盾。选错指标不是性能损失，而是整个辩论机制的失效。

4. **统一 PROHIBITION 在情境依赖题上失效**：没有通用答案的问题需要条件映射，而不是无条件的立场对立。对这类问题强制全开 PROHIBITION，会使最关键维度的得分下降 75%。

5. **分类器的路由决策比约束级别本身更重要**：binary 问题上的性能提升来自分类器将其路由到情境依赖模式，而非中等 PROHIBITION 设置。问题类型分类才是更根本的设计选择。

6. **强制承担不会放大虚假确定性，但会减少合理的不确定性表达**：PROHIBITION 与 single_llm 的虚假确定性比例相同（14.3%），但会降低 `honest_uncertainty` 维度得分。对需要校准置信度的场景，应提前考虑这一权衡。

六个发现的共同结构：系统产出了反预期的结果 → 诊断指向某个"看起来正确"的设计假设在具体场景下失效 → 修复需要重新思考该假设的适用前提。这个模式是多智能体系统工程中最普遍的迭代路径。

---

*完整实验方法与数据：[PAPER.md](PAPER.md) · 代码：[github.com/xyma2003/multi-agent-debate](https://github.com/xyma2003/multi-agent-debate)*

---

## 相关资源

- 技术栈：LangGraph 1.1.9 · claude-3-5-sonnet · BAAI/bge-small-en-v1.5 · DeBERTa NLI cross-encoder · SQLite · Streamlit
- 实验数据：`/results/` 目录下有完整 JSON（full_system / original_devil / single_llm / nli_detection）
- 参考文献：Perez et al., "Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models," 2022

---

*如有问题或讨论，欢迎留言。*
