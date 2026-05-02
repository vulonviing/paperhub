# PaperHub

PaperHub fetches HuggingFace Daily Papers by natural-language or programmatic
date filters, assigns each paper to its own AI summarization agent, and
renders English, Jupyter-friendly summaries by default. Turkish output is
available with `language="tr"` or `--language tr`.

```python
from paperhub import PaperHub

PaperHub().run("may 2026 top 10 papers")
```

## Install

```bash
python -m pip install -e ".[dev]"
```

Optional provider extras:

```bash
python -m pip install -e ".[openai]"     # adds the OpenAI client
python -m pip install -e ".[google]"     # adds the Google Gemini client
```

## Environment Variables

PaperHub reads configuration from environment variables (or a `.env` file at
the project root). Copy `.env.example` to `.env` to start:

```bash
cp .env.example .env
```

| Variable                       | Purpose                                       |
|--------------------------------|-----------------------------------------------|
| `ANTHROPIC_API_KEY`            | Default provider key                          |
| `OPENAI_API_KEY`               | Optional, used when `--provider openai`       |
| `GOOGLE_API_KEY`               | Optional, used when `--provider google`       |
| `PAPERHUB_PROVIDER`            | Override default provider (default: anthropic)|
| `PAPERHUB_MODEL`               | Optional global model override                |
| `PAPERHUB_ANTHROPIC_MODEL`     | Anthropic default model (default: claude-3-5-haiku-20241022)|
| `PAPERHUB_OPENAI_MODEL`        | OpenAI default model (default: gpt-4o-mini)   |
| `PAPERHUB_GOOGLE_MODEL`        | Google default model (default: gemini-2.5-pro)|
| `PAPERHUB_CONCURRENCY`         | Max concurrent paper agents (default: 5)      |
| `PAPERHUB_MAX_PDF_CHARS`       | Truncation cap for PDF text (default: 60000)  |
| `PAPERHUB_CACHE_DIR`           | Override the on-disk cache location           |

## Quickstart

### Notebook

For a step-by-step notebook that covers installation, API keys, readiness
checks, metadata smoke tests, and the first live `PaperHub.run(...)` call, open
`examples/03_jupyter_quickstart.ipynb`.

```python
from paperhub import PaperHub

hub = PaperHub()
hub.run("may 2026 top 10 papers")
```

The notebook cell renders one section per paper with motivation, method,
findings, concrete real-world examples, and a summary capped at 6000
characters. Turkish readers can open
`examples/03_jupyter_quickstart_tr.ipynb`.

### Turkish Output

```python
from paperhub import PaperHub

hub = PaperHub(language="tr")
hub.run("mayis 2026 top 10 paper")
```

You can also choose Turkish per call:

```python
hub.run("today top 5", language="tr")
```

### Programmatic

```python
from datetime import date
from paperhub import PaperHub

hub = PaperHub()
hub.run(period="month", year=2026, month=5, top_n=10)
hub.run(period="week", year=2026, week=18, top_n=5)
hub.run(period="custom", start=date(2026, 4, 15), end=date(2026, 4, 30), top_n=15)
```

`run` returns a `list[PaperSummary]`; in non-Jupyter contexts pass
`display=False` and call `render_plain` yourself if you do not need the
Markdown side effect.

### CLI

```bash
paperhub "may 2026 top 10 papers"
paperhub "this month top 5"
paperhub "2026-05-01 to 2026-05-31 top 5"
paperhub --provider openai --model gpt-4o-mini "may 2026 top 3"
paperhub --language tr "mayis 2026 top 3"
paperhub --no-summarize "may 2026 top 5"   # metadata only, no LLM call
```

If the selected provider's API key is missing, the CLI prints a clear message
and exits with code 3 instead of crashing. Use `--no-summarize` to verify the
fetch pipeline without an LLM key.

## Provider Selection

```python
PaperHub(model="claude-3-5-haiku-20241022") # Anthropic Haiku (default)
PaperHub(provider="openai")                 # Uses OpenAI default model
PaperHub(model="gpt-4o-mini", provider="openai")
PaperHub(model="gemini-2.5-pro", provider="google")
```

If `model` is omitted, PaperHub picks the selected provider's default model.
Provider-specific `.env` values such as `PAPERHUB_OPENAI_MODEL` override those
defaults. `PAPERHUB_MODEL` remains available as a global override, but a model
id that clearly belongs to another provider is ignored for the selected
provider.

Provider SDKs are imported lazily — installing `paperhub` does not require
the OpenAI or Google packages unless you actually use those providers.

## Caching

PaperHub caches metadata, PDF text, and summaries in
`~/.cache/paperhub/paperhub.sqlite` (override with `PAPERHUB_CACHE_DIR`).
Summary entries are keyed by `(arxiv_id, model, language)`, so swapping models
or output language gives you a clean re-run while keeping the PDF download
free.

A second invocation with the same papers and model:

- Reuses the cached PDF text (no arXiv hit, no extraction).
- Reuses the cached summary (no LLM call).

## Tests, Lint, Typecheck, Build

```bash
python -m pytest          # unit tests, no live network or LLM keys needed
python -m ruff check .
python -m ruff format --check .
python -m mypy src tests
python -m build           # build wheel + sdist
```

The unit tests use a fake LLM client and `httpx.MockTransport`; no real
provider keys are required.

## Repository Scope

The repository is kept to the files needed to install, use, test, and
understand the package:

- `src/paperhub/` for runtime package code.
- `tests/` for the mocked test suite.
- `docs/ARCHITECTURE.md` for implementation notes.
- `examples/` for clean, re-runnable examples.
- `.env.example` as a visible environment template.

Local secrets, build outputs, caches, downloaded PDFs, maintainer-only notes,
and one-off project scaffolding are excluded with `.gitignore`.

## Project Layout

```
src/paperhub/
  __init__.py           # PaperHub public API
  config.py             # Settings (pydantic-settings)
  models.py             # PaperMeta, PaperSummary, RunRequest
  dates.py              # period → (start, end)
  nl_parser.py          # English/Turkish NL → RunRequest
  fetchers/             # HF JSON API (default) + HTML fallback
  pdf/                  # arXiv download + text extraction
  agents/               # LLMClient protocol + provider clients + PaperAgent
  orchestrator.py       # asyncio.Semaphore parallelism
  cache.py              # SQLite cache
  formatter.py          # Markdown / plain-text rendering
  cli.py                # `paperhub` entrypoint
tests/                  # pytest suite, mocked HTTP and fake LLM
docs/ARCHITECTURE.md
examples/01_quickstart.py
examples/02_programmatic.py
examples/03_jupyter_quickstart.ipynb
examples/03_jupyter_quickstart_tr.ipynb
```

## Troubleshooting

- *“Could not parse query”*: try ASCII forms (`mayis 2026 top 10`) or an ISO
  date (`2026-05-15`).
- *“No papers found”*: HuggingFace may not yet have published Daily Papers
  for that date. Try `--no-summarize` to confirm that the fetcher is hitting
  the API at all.
- *PDF text comes back tiny*: some arXiv PDFs use unusual layouts. PaperHub
  falls back to `pdfplumber`; if both extractors are short, the agent will
  pass through the abstract as the input text instead of failing.
