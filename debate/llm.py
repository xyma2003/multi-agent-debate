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

  cerebras:
    CEREBRAS_API_KEY=csk_...
    CEREBRAS_MODEL=llama-3.3-70b  (optional)
    Base URL: https://api.cerebras.ai/v1 (OpenAI-compatible)

  together:
    TOGETHER_API_KEY=...
    TOGETHER_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo  (optional)
    Base URL: https://api.together.xyz/v1 (OpenAI-compatible)

  sambanova:
    SAMBANOVA_API_KEY=...
    SAMBANOVA_MODEL=Meta-Llama-3.3-70B-Instruct  (optional)
    Base URL: https://api.sambanova.ai/v1 (OpenAI-compatible)

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

    elif backend == "cerebras":
        from langchain_openai import ChatOpenAI
        model = os.environ.get("CEREBRAS_MODEL", "llama-3.3-70b")
        return ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=os.environ["CEREBRAS_API_KEY"],
            openai_api_base="https://api.cerebras.ai/v1",
        )

    elif backend == "together":
        from langchain_openai import ChatOpenAI
        model = os.environ.get("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
        return ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=os.environ["TOGETHER_API_KEY"],
            openai_api_base="https://api.together.xyz/v1",
        )

    elif backend == "sambanova":
        from langchain_openai import ChatOpenAI
        model = os.environ.get("SAMBANOVA_MODEL", "Meta-Llama-3.3-70B-Instruct")
        return ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=os.environ["SAMBANOVA_API_KEY"],
            openai_api_base="https://api.sambanova.ai/v1",
        )

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
