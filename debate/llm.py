# debate/llm.py
"""
Centralized LLM factory for all debate agent nodes.

Select backend via LLM_BACKEND env var (default: groq):

  groq (default):
    GROQ_API_KEY=gsk_...
    GROQ_MODEL=llama-3.3-70b-versatile  (optional)

  qwen:
    GROQ_API_KEY=gsk_...   (Qwen is hosted on Groq)
    QWEN_MODEL=qwen/qwen3-32b  (optional)

  anthropic:
    ANTHROPIC_API_KEY=sk-ant-...
    ANTHROPIC_CUSTOM_HEADERS — newline-separated "Key: Value" for proxy auth

  openai:
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4o-mini  (optional)
"""
import os


def _make_llm():
    """Return a LangChain chat model. Backend selected by LLM_BACKEND env var at call time."""
    backend = os.environ.get("LLM_BACKEND", "groq").lower()

    if backend == "groq":
        from langchain_groq import ChatGroq
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(model=model, temperature=0)

    elif backend == "qwen":
        from langchain_groq import ChatGroq
        model = os.environ.get("QWEN_MODEL", "qwen/qwen3-32b")
        return ChatGroq(model=model, temperature=0)

    elif backend == "anthropic":
        from langchain_anthropic import ChatAnthropic
        custom_headers_str = os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "")
        headers: dict[str, str] = {}
        if custom_headers_str:
            for line in custom_headers_str.split("\n"):
                line = line.strip()
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip()] = v.strip()
        return ChatAnthropic(model="claude-sonnet-4-6", default_headers=headers)

    elif backend == "openai":
        from langchain_openai import ChatOpenAI
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0)

    else:  # gemini
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=0)
