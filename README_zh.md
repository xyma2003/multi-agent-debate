# 多智能体辩论系统 (Multi-Agent Debate System)

基于 LangGraph 构建的多智能体系统。该系统包含三个具有不同认知偏差的 LLM 智能体——**乐观主义者 (Optimist)**、**悲观主义者 (Pessimist)** 和 **恶魔代言人 (Devil's Advocate)**。它们通过多轮结构化辩论对任何话题进行深入探讨。系统能够检测真实的语义分歧，追踪带有归因的让步情况，并最终生成一份具有公式推导置信度的、可审计的共识报告。

作为一个展示项目，它演示了：LangGraph 多智能体图、语义分歧检测、Pydantic 结构化输出、SQLite 持久化以及 Streamlit 流式 UI。

---

## 演示 (Demo)

```text
User: "远程办公对公司有净收益吗？"

第一轮 (并行):
  🟢 乐观主义者    → "远程办公将生产力提高了 15-20%..."
  🔴 悲观主义者   → "团队协作和企业文化受到了不可挽回的损害..."
  😈 恶魔代言人 → "这种生产力的提升可能存在幸存者偏差..."

分歧得分: 0.82 → 触发第二轮

第二轮 (反驳):
  🟢 乐观主义者    → 让步: "对初级员工来说，企业文化方面的风险确实存在"
  🔴 悲观主义者   → 坚持原有立场
  😈 恶魔代言人 → 转变立场: "混合办公才是真正的最优解"

最终报告:
  置信度: 71% | 状态: 已达成共识 (Converged)
  共识: ["异步沟通工具是必不可少的", ...]
  争议点: [{"topic": "文化影响", "optimist": "...", "pessimist": "..."}]
```

---

## 前置要求

- **Python 3.10+**
- **Anthropic API 访问权限** — 可以是直接的 API Key 或者通过代理访问（见下文）

---

## 安装与运行 (Setup)

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/debate-agent.git
cd debate-agent
```

### 2. 创建虚拟环境

```bash
# 选项 A: 使用内置的 venv
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 选项 B: 使用 conda
conda create -n debate-agent python=3.10
conda activate debate-agent
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意：** 首次运行时会自动从 HuggingFace 下载 `BAAI/bge-small-en-v1.5` 嵌入模型（约 130MB）。这会在首次发起辩论时自动触发。

### 4. 配置 API 密钥

复制示例环境变量文件并填入你的凭据：

```bash
cp .env.example .env
```

**选项 A — 直接使用 Anthropic API key（标准方式）：**

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-api03-...
```

**选项 B — 使用内部代理（如公司代理）：**

```bash
# .env
ANTHROPIC_BASE_URL=https://your-proxy-base-url
ANTHROPIC_AUTH_TOKEN=your-auth-token
ANTHROPIC_CUSTOM_HEADERS=X-Custom-Header: value
```

> 应用程序会根据设置的环境变量自动检测使用哪种认证方式，无需修改代码。

如果你没有使用自动加载 `.env` 的工具，可以在运行前将其加载到环境变量中：

```bash
# macOS/Linux — 可以在启动 streamlit 前运行：
export $(grep -v '^#' .env | xargs)
```
> 或者应用程序也会自动使用 `python-dotenv` 加载 `.env` 文件。

### 5. 运行

```bash
streamlit run app.py
```

在浏览器中打开 **http://localhost:8501** 即可访问 UI 界面。

---

## 使用指南 (Usage)

1. 输入任何主题或问题（例如：*“AI 监管对创新有好处吗？”*）
2. 设置 **最大辩论轮数 (Max Rounds)** (1–3 轮) — 轮数越多 = 反驳周期越多，探讨越深入
3. 点击 **Start Debate** — 即可实时观看智能体之间的辩论
4. 阅读最终报告：包含置信度得分、最终结论、共识/争议点分类以及各自的推理追踪记录
5. 历史辩论记录会显示在 **侧边栏 (sidebar)**，可以即时回放而无需重新运行智能体

---

## 工作原理 (How It Works)

```text
User topic
    │
    ▼
initialize ──► [Optimist | Pessimist | Devil's Advocate]  (第一轮, 并行)
                    │
                    ▼
            collect_round1
                    │
                    ▼
     divergence_check_node  ← 基于 key_claims 嵌入向量的语义相似度计算
                    │
          ┌─────────┴─────────┐
       有分歧 (diverged)     达成共识 (converged) / 达到最大轮数
          │                    │
   [反驳轮次 (rebuttal)]     synthesize_stub
          │                    │
      (循环返回)            save_node → 存入 SQLite
                               │
                           DebateReport (生成最终报告)
```
