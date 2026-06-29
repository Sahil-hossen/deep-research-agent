"""Streamlit UI for the Deep Research Agent.

Type a question, watch the planner -> researchers -> writer -> critic work in
real time, and read the final cited report.
"""

from __future__ import annotations

import streamlit as st

from research_agent.config import settings
from research_agent.llm import LLMError
from research_agent.models import Critique, Finding
from research_agent.orchestrator import run_research

st.set_page_config(page_title="Deep Research Agent", page_icon="🔎", layout="centered")

EXAMPLES = [
    "Compare the EV battery strategies of Tesla, BYD, and Toyota.",
    "What are the main pros and cons of the RAG vs fine-tuning approach for LLMs?",
    "How does the Raft consensus algorithm work, and where is it used?",
]


def sidebar() -> None:
    with st.sidebar:
        st.header("About")
        st.markdown(
            "A **multi-agent** research system built from scratch:\n\n"
            "1. **Planner** splits the question\n"
            "2. **Researchers** search the web\n"
            "3. **Writer** drafts a cited report\n"
            "4. **Critic** reviews & triggers a revision *(Reflexion)*"
        )
        st.divider()
        st.caption("Provider")
        st.code(f"{settings.provider}  ·  {settings.model}", language="text")
        st.caption(
            f"Sub-questions: {settings.max_sub_questions} · "
            f"Results/search: {settings.max_search_results} · "
            f"Max revisions: {settings.max_revisions}"
        )


def render_finding(finding: Finding) -> None:
    st.markdown(f"**{finding.sub_question}**")
    st.write(finding.summary)
    if finding.sources:
        with st.expander(f"{len(finding.sources)} sources"):
            for s in finding.sources:
                st.markdown(f"- [{s.title}]({s.url})")


def render_critique(verdict: Critique) -> None:
    st.markdown(f"**Quality score:** {verdict.score}/10")
    if verdict.issues:
        st.markdown("**Issues raised:**")
        for issue in verdict.issues:
            st.markdown(f"- {issue}")


def run(query: str) -> None:
    final_report: str | None = None

    with st.status("Researching...", expanded=True) as status:
        try:
            for event in run_research(query):
                if event.stage == "plan" and isinstance(event.data, list):
                    st.write("**Plan — sub-questions:**")
                    for q in event.data:
                        st.markdown(f"- {q}")
                elif event.stage == "research" and isinstance(event.data, Finding):
                    render_finding(event.data)
                elif event.stage == "critique" and isinstance(event.data, Critique):
                    render_critique(event.data)
                elif event.stage in {"draft", "revise"} and isinstance(event.data, str):
                    st.write(f"_{event.message}_")
                elif event.stage == "done":
                    final_report = event.data  # type: ignore[assignment]
                else:
                    st.write(event.message)
            status.update(label="Research complete", state="complete")
        except LLMError as exc:
            status.update(label="Failed", state="error")
            st.error(str(exc))
            return

    if final_report:
        st.markdown("## Report")
        st.markdown(final_report)
        st.download_button(
            "Download report (.md)",
            data=final_report,
            file_name="research_report.md",
            mime="text/markdown",
        )


def main() -> None:
    st.title("🔎 Deep Research Agent")
    st.caption("A multi-agent system that researches the web and writes a cited report.")
    sidebar()

    st.write("**Try an example:**")
    cols = st.columns(len(EXAMPLES))
    picked: str | None = None
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, use_container_width=True):
            picked = example

    typed = st.chat_input("Ask a research question...")
    query = picked or typed
    if query:
        st.markdown(f"### {query}")
        run(query)


if __name__ == "__main__":
    main()
