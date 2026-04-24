# debate/llm.py
"""
Centralized LLM factory for all debate agent nodes.

The Meituan internal proxy requires three env vars:
  ANTHROPIC_BASE_URL     — proxy endpoint (read automatically by anthropic SDK)
  ANTHROPIC_AUTH_TOKEN   — auth token (read automatically by anthropic SDK as api_key)
  ANTHROPIC_CUSTOM_HEADERS — newline-separated "Key: Value" pairs passed as default_headers

No ANTHROPIC_API_KEY is needed or used.
"""
import os

from langchain_anthropic import ChatAnthropic

MODEL_ID = "claude-sonnet-4-6"  # Locked in CLAUDE.md. Do NOT change to claude-sonnet-4-5.


def _make_llm() -> ChatAnthropic:
    """Return ChatAnthropic configured for the Meituan internal proxy.

    Reads ANTHROPIC_CUSTOM_HEADERS as newline-separated 'Key: Value' pairs.
    ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN are picked up automatically
    by the underlying anthropic SDK from the environment.
    """
    custom_headers_str = os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "")
    headers: dict[str, str] = {}
    if custom_headers_str:
        for line in custom_headers_str.split("\n"):
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip()] = v.strip()
    return ChatAnthropic(model=MODEL_ID, default_headers=headers)
