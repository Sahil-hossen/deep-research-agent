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


def load_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()

    if provider == "ollama":
        return Settings(
            provider="ollama",
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            # Ollama ignores the key but the OpenAI SDK requires a non-empty value.
            api_key="ollama",
        )

    # Default: Groq free tier.
    return Settings(
        provider="groq",
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        base_url=GROQ_BASE_URL,
        api_key=os.getenv("GROQ_API_KEY", ""),
    )


settings = load_settings()
