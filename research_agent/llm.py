"""Thin LLM wrapper.

Both Groq and Ollama expose an OpenAI-compatible API, so a single client and
helper covers both providers. This is intentionally minimal — the interesting
logic lives in the agents, not here.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from openai import OpenAI

from .config import settings


class LLMError(RuntimeError):
    """Raised when the LLM call or response parsing fails."""


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if settings.provider == "groq" and not settings.api_key:
        raise LLMError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add a free "
            "key from https://console.groq.com/keys (or switch LLM_PROVIDER to "
            "'ollama' for fully offline use)."
        )
    return OpenAI(base_url=settings.base_url, api_key=settings.api_key)


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> str:
    """Send a chat-completion request and return the assistant text."""
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = get_client().chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 - surface provider errors uniformly
        raise LLMError(f"LLM request failed: {exc}") from exc

    content = response.choices[0].message.content
    if not content:
        raise LLMError("LLM returned an empty response.")
    return content.strip()


def chat_json(messages: list[dict[str, str]], *, temperature: float = 0.2) -> Any:
    """Chat helper that parses the response as JSON, with a salvage fallback."""
    raw = chat(messages, temperature=temperature, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Some models wrap JSON in prose or code fences; salvage the object.
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError as exc:
                raise LLMError(f"Could not parse JSON from response: {raw[:200]}") from exc
        raise LLMError(f"Expected JSON but got: {raw[:200]}")
