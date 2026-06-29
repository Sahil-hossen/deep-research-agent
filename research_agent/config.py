"""Central configuration loaded from environment variables.

Keeps the rest of the codebase provider-agnostic: switch between Groq (free
hosted) and a local Ollama install by changing only the .env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    base_url: str
    api_key: str

    # Pipeline guardrails
    max_sub_questions: int = 4
    max_search_results: int = 5
    max_revisions: int = 1

    # Skip TLS verification — only needed on corporate networks that do SSL
    # inspection with a self-signed root certificate. Off by default.
    disable_ssl_verify: bool = False


def _env_flag(name: str) -> bool:
    return os.getenv(name, "false").lower().strip() in {"1", "true", "yes"}


def load_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    disable_ssl_verify = _env_flag("DISABLE_SSL_VERIFY")

    if provider == "ollama":
        return Settings(
            provider="ollama",
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            # Ollama ignores the key but the OpenAI SDK requires a non-empty value.
            api_key="ollama",
            disable_ssl_verify=disable_ssl_verify,
        )

    # Default: Groq free tier.
    return Settings(
        provider="groq",
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        base_url=GROQ_BASE_URL,
        api_key=os.getenv("GROQ_API_KEY", ""),
        disable_ssl_verify=disable_ssl_verify,
    )


settings = load_settings()
