"""The four agents that make up the research team.

Each agent is just a function that takes some input, calls the LLM with a
focused prompt, and returns a typed result. Reading this file top-to-bottom
shows the whole pipeline: plan -> research -> write -> critique.
"""

from __future__ import annotations

from .llm import chat, chat_json
from .models import Critique, Finding, Source
from .tools import web_search


# ---------------------------------------------------------------------------
# 1. Planner — breaks a broad question into focused sub-questions.
# ---------------------------------------------------------------------------
def plan(query: str, max_sub_questions: int = 4) -> list[str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research planner. Break the user's question into "
                f"{max_sub_questions} or fewer focused, non-overlapping "
                "sub-questions that together fully answer it. "
                'Reply as JSON: {"sub_questions": ["...", "..."]}'
            ),
        },
        {"role": "user", "content": query},
    ]
    data = chat_json(messages)
    sub_questions = data.get("sub_questions", [])
    # Keep it safe: drop empties and respect the limit.
    cleaned = [q.strip() for q in sub_questions if q and q.strip()]
    return cleaned[:max_sub_questions] or [query]


# ---------------------------------------------------------------------------
# 2. Researcher — searches the web for one sub-question and summarises it.
# ---------------------------------------------------------------------------
def research(sub_question: str, max_results: int = 5) -> Finding:
    sources = web_search(sub_question, max_results=max_results)

    if not sources:
        return Finding(
            sub_question=sub_question,
            summary="No web results were found for this sub-question.",
            sources=[],
        )

    # Give the model the raw search snippets and ask for a grounded summary.
    context = "\n\n".join(
        f"[{i + 1}] {s.title}\n{s.snippet}\nURL: {s.url}"
        for i, s in enumerate(sources)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a researcher. Summarise the search results to answer "
                "the sub-question. Use only the information given, stay factual, "
                "and do not invent details. 3-5 sentences."
            ),
        },
        {
            "role": "user",
            "content": f"Sub-question: {sub_question}\n\nSearch results:\n{context}",
        },
    ]
    summary = chat(messages, temperature=0.2)
    return Finding(sub_question=sub_question, summary=summary, sources=sources)


# ---------------------------------------------------------------------------
# 3. Writer — synthesises all findings into one cited markdown report.
# ---------------------------------------------------------------------------
def write_report(
    query: str,
    findings: list[Finding],
    feedback: list[str] | None = None,
) -> str:
    findings_block = "\n\n".join(
        f"Sub-question: {f.sub_question}\nFindings: {f.summary}"
        for f in findings
    )

    feedback_block = ""
    if feedback:
        issues = "\n".join(f"- {item}" for item in feedback)
        feedback_block = (
            "\n\nA reviewer raised these issues with your previous draft. "
            f"Fix them in this version:\n{issues}"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research writer. Using the findings below, write a "
                "clear, well-structured report in markdown that answers the "
                "user's question. Use headings and short paragraphs. Base every "
                "claim on the findings; do not add outside information."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\nFindings:\n{findings_block}{feedback_block}"
            ),
        },
    ]
    return chat(messages, temperature=0.4)


# ---------------------------------------------------------------------------
# 4. Critic — reviews the draft and decides if a revision is needed (Reflexion).
# ---------------------------------------------------------------------------
def critique(query: str, report: str) -> Critique:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a critical reviewer. Judge whether the report fully and "
                "accurately answers the question. Look for missing angles, vague "
                "claims, or contradictions. Reply as JSON: "
                '{"needs_revision": true/false, "score": 1-10, "issues": ["..."]}'
            ),
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nReport:\n{report}",
        },
    ]
    data = chat_json(messages)
    return Critique(
        needs_revision=bool(data.get("needs_revision", False)),
        score=int(data.get("score", 0) or 0),
        issues=[str(i) for i in data.get("issues", [])],
    )


def collect_sources(findings: list[Finding]) -> list[Source]:
    """De-duplicate sources across findings by URL, preserving order."""
    seen: set[str] = set()
    unique: list[Source] = []
    for f in findings:
        for s in f.sources:
            if s.url and s.url not in seen:
                seen.add(s.url)
                unique.append(s)
    return unique
