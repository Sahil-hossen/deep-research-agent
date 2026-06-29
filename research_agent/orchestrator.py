"""Runs the multi-agent pipeline and streams progress as it goes.

`run_research` is a generator: it `yield`s an Event after each step so the UI
can show the agents working in real time. The final event holds the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from . import agents
from .config import settings
from .models import Finding, Source


@dataclass
class Event:
    """A single step in the pipeline, sent to the UI as it happens."""

    stage: str  # "plan" | "research" | "draft" | "critique" | "revise" | "done"
    message: str
    data: object | None = None


def run_research(query: str) -> Iterator[Event]:
    # 1. Plan
    yield Event("plan", "Planning the research...")
    sub_questions = agents.plan(query, settings.max_sub_questions)
    yield Event("plan", f"Created {len(sub_questions)} sub-questions.", sub_questions)

    # 2. Research each sub-question
    findings: list[Finding] = []
    for i, sq in enumerate(sub_questions, start=1):
        yield Event("research", f"Researching ({i}/{len(sub_questions)}): {sq}")
        finding = agents.research(sq, settings.max_search_results)
        findings.append(finding)
        yield Event("research", f"Done: {sq}", finding)

    # 3. Write the first draft
    yield Event("draft", "Writing the report...")
    report = agents.write_report(query, findings)
    yield Event("draft", "First draft ready.", report)

    # 4. Critique + optional revision (the Reflexion loop)
    for round_no in range(settings.max_revisions):
        yield Event("critique", "Reviewing the draft for gaps...")
        verdict = agents.critique(query, report)
        yield Event("critique", f"Quality score: {verdict.score}/10", verdict)

        if not verdict.needs_revision:
            break

        yield Event("revise", "Revising based on the critique...")
        report = agents.write_report(query, findings, feedback=verdict.issues)
        yield Event("revise", "Revised draft ready.", report)

    # 5. Attach a references section and finish
    sources = agents.collect_sources(findings)
    final_report = _append_references(report, sources)
    yield Event("done", "Research complete.", final_report)


def _append_references(report: str, sources: list[Source]) -> str:
    if not sources:
        return report
    lines = ["\n\n## References"]
    for i, s in enumerate(sources, start=1):
        lines.append(f"{i}. [{s.title}]({s.url})")
    return report + "\n".join(lines)
