"""Tools the agents can use to gather information.

Currently just web search via DuckDuckGo, which is free and needs no API key.
Kept separate from the agents so new tools are easy to add later.
"""

from __future__ import annotations

from ddgs import DDGS

from .config import settings
from .models import Source


def web_search(query: str, max_results: int = 5) -> list[Source]:
    """Search the web and return a list of Source objects.

    Never raises: on any failure it returns an empty list so the pipeline can
    keep going instead of crashing on one bad query.
    """
    try:
        # verify=False is only used on corporate networks with SSL inspection.
        with DDGS(verify=not settings.disable_ssl_verify) as ddgs:
            results = ddgs.text(query, max_results=max_results)
    except Exception:  # noqa: BLE001 - network/rate-limit issues shouldn't crash a run
        return []

    sources: list[Source] = []
    for r in results:
        sources.append(
            Source(
                title=r.get("title", "Untitled"),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
            )
        )
    return sources
