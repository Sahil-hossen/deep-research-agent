# Deep Research Agent

A **multi-agent research system** built from scratch — give it a question and a
team of AI agents plans the research, searches the web, writes a cited report,
and critiques its own work before finalizing. Think of it as a tiny,
self-hostable "Deep Research" / Perplexity-style tool.

100% free to run: uses [Groq](https://groq.com)'s free API tier (or a local
[Ollama](https://ollama.com) model) and DuckDuckGo web search — **no paid keys**.

---

## What it does

You ask something broad, e.g. *"Compare the EV battery strategies of Tesla, BYD,
and Toyota."* The agents collaborate:

| Agent | Role |
|-------|------|
| **Planner** | Breaks the question into focused sub-questions |
| **Researcher** | Searches the web for each sub-question and summarizes findings |
| **Writer** | Synthesizes all findings into a structured markdown report |
| **Critic** | Reviews the draft for gaps/vagueness and triggers a revision *(Reflexion pattern)* |

You watch the whole process live in the UI and get a downloadable, cited report.

## Architecture

```
                +----------+
   question --> | Planner  | --> sub-questions
                +----------+
                     |
                     v  (for each sub-question)
                +----------+     +-------------+
                |Researcher| <-> | web_search  |  (DuckDuckGo, free)
                +----------+     +-------------+
                     | findings
                     v
                +----------+
                |  Writer  | --> draft report
                +----------+
                     |
                     v
                +----------+   needs revision?
                |  Critic  | ---- yes --> back to Writer (max 1 round)
                +----------+
                     | no
                     v
              final cited report
```

The orchestrator (`research_agent/orchestrator.py`) is a generator that `yield`s
a progress event after every step, which is what powers the live UI.

## Tech stack

- **Python 3.13**
- **Groq** free API (default model `llama-3.3-70b-versatile`), swappable to **Ollama**
- **OpenAI Python SDK** — works for both providers (both are OpenAI-compatible)
- **DuckDuckGo** (`ddgs`) for free, key-less web search
- **Streamlit** for the UI
- **pytest** for tests

## Design decisions

- **No agent framework (LangChain/LangGraph) on purpose.** The orchestration is
  hand-written so the logic — tool routing, the research loop, and the
  self-critique (Reflexion) cycle — is fully visible and easy to explain. For a
  production system with complex branching, checkpointing, or human-in-the-loop,
  a framework like LangGraph would be the right call.
- **Provider-agnostic.** Switching from free hosted (Groq) to fully offline
  (Ollama) is a one-line change in `.env`.
- **Fails soft.** Web-search errors return no results instead of crashing a run.

## Project structure

```
deep-research-agent/
  app.py                      # Streamlit UI
  research_agent/
    config.py                 # settings loaded from .env
    llm.py                    # tiny LLM helper (Groq / Ollama)
    tools.py                  # web search (DuckDuckGo)
    models.py                 # small data classes
    agents.py                 # the 4 agents: planner/researcher/writer/critic
    orchestrator.py           # runs the pipeline, streams progress
  tests/                      # unit tests (run offline, no API key)
  requirements.txt
  .env.example
```

## Quickstart

```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your free Groq key
copy .env.example .env            # Windows  (cp on macOS/Linux)
# then edit .env and paste your key from https://console.groq.com/keys

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Run fully offline (no API key)

Install [Ollama](https://ollama.com), pull a model, and set these in `.env`:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
```

## Corporate networks (SSL inspection)

If you see `CERTIFICATE_VERIFY_FAILED` errors, you're likely behind a corporate
proxy that inspects TLS with a self-signed root certificate. Set
`DISABLE_SSL_VERIFY=true` in your `.env` to skip verification. Leave it `false`
everywhere else.

## Tests

```bash
pytest -q
```

Tests mock the LLM and web search, so they run offline and for free.

## Possible extensions

- Add more tools (Wikipedia, a calculator, a PDF reader) to the `tools` module
- Parallelize the researchers with `asyncio` for faster runs
- Add a retrieval step over your own documents (Agentic RAG)
- Stream tokens from the writer for a typewriter effect

---

Built by [Sahil Hossen](https://github.com/Sahil-hossen).
