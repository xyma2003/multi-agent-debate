"""Pytest configuration — shared fixtures and integration marker registration."""
from __future__ import annotations

import os

import pytest


def pytest_configure(config):
    """Register the integration marker to silence PytestUnknownMarkWarning."""
    config.addinivalue_line("markers", "integration: requires live LLM backend")


# Map LLM_BACKEND → required env var
_BACKEND_CRED_MAP = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
}


@pytest.fixture(autouse=True)
def skip_live_llm_without_credentials(request):
    """Auto-skip @pytest.mark.integration tests when the active backend has no creds."""
    if not request.node.get_closest_marker("integration"):
        return
    backend = os.environ.get("LLM_BACKEND", "groq").lower()
    if backend == "anthropic":
        if not (
            os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or os.environ.get("ANTHROPIC_API_KEY")
        ):
            pytest.skip(f"LLM_BACKEND=anthropic but no ANTHROPIC_* credentials")
    else:
        env_var = _BACKEND_CRED_MAP.get(backend)
        if env_var and not os.environ.get(env_var):
            pytest.skip(f"LLM_BACKEND={backend} but no {env_var}")
