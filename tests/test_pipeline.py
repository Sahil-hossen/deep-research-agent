"""Unit tests for the pure (non-network) parts of the pipeline.

These run without an API key or internet because they test the logic that
stitches agent outputs together, mocking the LLM and web-search calls.
"""

from research_agent import agents
from research_agent.models import Finding, Source
from research_agent.orchestrator import _append_references


def test_collect_sources_dedupes_by_url():
    findings = [
        Finding("q1", "s1", [Source("A", "http://a.com", "..."),
                             Source("B", "http://b.com", "...")]),
        Finding("q2", "s2", [Source("A again", "http://a.com", "..."),
                             Source("C", "http://c.com", "...")]),
    ]
    sources = agents.collect_sources(findings)
    urls = [s.url for s in sources]
    assert urls == ["http://a.com", "http://b.com", "http://c.com"]


def test_append_references_numbers_sources():
    report = "# Report\nBody."
    sources = [Source("First", "http://1.com", ""), Source("Second", "http://2.com", "")]
    out = _append_references(report, sources)
    assert "## References" in out
    assert "1. [First](http://1.com)" in out
    assert "2. [Second](http://2.com)" in out


def test_append_references_noop_when_empty():
    report = "# Report"
    assert _append_references(report, []) == report


def test_plan_cleans_and_limits(monkeypatch):
    # Mock the LLM so the test is fast, free, and offline.
    monkeypatch.setattr(
        agents,
        "chat_json",
        lambda *a, **k: {"sub_questions": ["  a  ", "", "b", "c", "d", "e"]},
    )
    result = agents.plan("anything", max_sub_questions=3)
    assert result == ["a", "b", "c"]


def test_plan_falls_back_to_query_when_empty(monkeypatch):
    monkeypatch.setattr(agents, "chat_json", lambda *a, **k: {"sub_questions": []})
    assert agents.plan("my question") == ["my question"]


def test_research_handles_no_sources(monkeypatch):
    monkeypatch.setattr(agents, "web_search", lambda *a, **k: [])
    finding = agents.research("unanswerable")
    assert finding.sources == []
    assert "No web results" in finding.summary
