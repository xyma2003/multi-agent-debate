# debate/llm.py
"""
Centralized LLM factory for all debate agent nodes.

Supports two auth modes (see README Option A / Option B):

  Option A — Direct Anthropic API key (standard):
    ANTHROPIC_API_KEY=sk-ant-...

  Option B — Corporate/internal proxy:
    ANTHROPIC_BASE_URL      — proxy endpoint (read automatically by anthropic SDK)
    ANTHROPIC_AUTH_TOKEN    — auth token (read automatically by anthropic SDK as api_key)
    ANTHROPIC_CUSTOM_HEADERS — newline-separated "Key: Value" pairs for any extra
                               headers required by the proxy (e.g. routing or auth headers)

  Example ANTHROPIC_CUSTOM_HEADERS value:
    X-Custom-Header: my-value
    X-Another-Header: another-value

No ANTHROPIC_API_KEY is needed when using Option B.
"""
import os

from langchain_anthropic import ChatAnthropic

MODEL_ID = "claude-sonnet-4-6"  # Locked in CLAUDE.md. Do NOT change to claude-sonnet-4-5.


def _make_llm() -> ChatAnthropic:
    """Return ChatAnthropic with optional custom headers for proxy environments.

    Reads ANTHROPIC_CUSTOM_HEADERS as newline-separated 'Key: Value' pairs and
    passes them as default_headers to ChatAnthropic. If the env var is not set,
    no extra headers are added (standard API key auth works as-is).

    ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN are picked up automatically
    by the underlying anthropic SDK from the environment (Option B proxy setup).
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
