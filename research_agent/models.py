"""Small data classes passed between the agents.

Using simple dataclasses (instead of dicts) makes the data flow easy to read:
you can see exactly what each agent produces and consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Source:
    """A single web source found during research."""

    title: str
    url: str
    snippet: str


@dataclass
class Finding:
    """What the researcher learned about one sub-question."""

    sub_question: str
    summary: str
    sources: list[Source] = field(default_factory=list)


@dataclass
class Critique:
    """The critic's verdict on a draft report."""

    needs_revision: bool
    score: int  # 1-10 quality score
    issues: list[str] = field(default_factory=list)
