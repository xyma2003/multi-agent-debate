# debate/llm.py
"""
Centralized LLM factory for all debate agent nodes.

Select backend via LLM_BACKEND env var (default: gemini):

  gemini (default):
    GOOGLE_API_KEY=...
    GEMINI_MODEL=gemini-1.5-flash  (optional)

  openai:
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4o-mini  (optional)

  anthropic:
    ANTHROPIC_API_KEY=sk-ant-...
    ANTHROPIC_CUSTOM_HEADERS — newline-separated "Key: Value" for proxy auth
"""
import os

_BACKEND = os.environ.get("LLM_BACKEND", "groq").lower()


def _make_llm():
    """Return a LangChain chat model. Backend selected by LLM_BACKEND env var."""
    if _BACKEND == "groq":
        from langchain_groq import ChatGroq
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(model=model, temperature=0)

    elif _BACKEND == "anthropic":
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

    elif _BACKEND == "openai":
        from langchain_openai import ChatOpenAI
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0)

    else:  # gemini (default)
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=0)
