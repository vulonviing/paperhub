# PaperHub

PaperHub fetches HuggingFace Daily Papers by programmatic date filters, assigns
each paper to its own AI summarization agent, and renders English, Jupyter-friendly
summaries by default. Turkish output is available with `language="tr"`.

Two ways to use PaperHub:

1. **Terminal CLI** (`paperhub`) — interactive launcher with a REPL interface. **Primary mode.**
2. **Python / Jupyter** — import `PaperHub` and call `hub.run(...)` directly.

## Install

```bash
python3 -m pip install -e ".[dev]"
```

Optional provider extras:

```bash
python3 -m pip install -e ".[anthropic]"  # adds the Anthropic client
python3 -m pip install -e ".[google]"     # adds the Google Gemini client
```

## Environment Variables

PaperHub reads configuration from environment variables (or a `.env` file at
the project root). Copy `.env.example` to `.env` to start:

```bash
cp .env.example .env
```

| Variable                       | Purpose                                       |
|--------------------------------|-----------------------------------------------|
| `OPENAI_API_KEY`               | Default provider key                          |
| `ANTHROPIC_API_KEY`            | Optional, used when `--provider anthropic`    |
| `GOOGLE_API_KEY`               | Optional, used when `--provider google`       |
| `PAPERHUB_PROVIDER`            | Override default provider (default: openai)   |
| `PAPERHUB_MODEL`               | Optional global model override                |
| `PAPERHUB_OPENAI_MODEL`        | OpenAI default model (default: gpt-4o-mini)   |
| `PAPERHUB_ANTHROPIC_MODEL`     | Anthropic default model (default: claude-3-5-haiku-20241022)|
| `PAPERHUB_GOOGLE_MODEL`        | Google default model (default: gemini-2.5-pro)|
| `PAPERHUB_CONCURRENCY`         | Max concurrent paper agents (default: 5)      |
| `PAPERHUB_MAX_PDF_CHARS`       | Truncation cap for PDF text (default: 60000)  |
| `PAPERHUB_CACHE_DIR`           | Override the on-disk cache location           |

## Terminal CLI (Primary)

Start the interactive launcher:

```bash
paperhub
```

The launcher opens a REPL with a status dashboard showing the current provider,
model, API key state, date range, and top paper count. Use commands to configure
and run:

```text
/provider
/provider openai
/model
/model gpt-4o-mini
/model default
/language
/date 2026-05
/date 2026-05-15
/date 2026-W18
/date 2026-05-01 2026-05-31
/top 5
/metadata
/run
/api-keys
/quit
```

### Date formats for `/date`

| Example                        | Period   | Description           |
|-------------------------------|----------|-----------------------|
| `/date 2026-05`               | month    | May 2026              |
| `/date 2026`                  | year     | Full year 2026        |
| `/date 2026-05-15`            | day      | Single day            |
| `/date 2026-W18`              | week     | ISO week 18 of 2026   |
| `/date 2026-05-01 2026-05-31` | custom   | Inclusive date range  |

`/metadata` fetches HuggingFace paper metadata only and does not call an LLM.
If the selected provider key is missing, `/run` prints setup guidance and the
launcher can render `docs/API_KEYS.md` with `/api-keys`.

You can also pass startup flags:

```bash
paperhub --provider anthropic
paperhub --model claude-3-5-haiku-20241022
paperhub --language tr
paperhub --top-n 10
```

## Python / Jupyter API

```python
from datetime import date
from paperhub import PaperHub

hub = PaperHub(provider="openai")

# Month
hub.run(period="month", year=2026, month=5, top_n=10)

# Single day
hub.run(period="day", year=2026, month=5, day=1, top_n=5)

# ISO week
hub.run(period="week", year=2026, week=18, top_n=5)

# Full year
hub.run(period="year", year=2026, top_n=20)

# Custom range
hub.run(period="custom", start=date(2026, 4, 15), end=date(2026, 4, 30), top_n=15)
```

`run` returns a `list[PaperSummary]`; in non-Jupyter contexts pass
`display=False` and call `render_plain` yourself if you do not need the
Markdown side effect.

### Turkish Output

```python
from paperhub import PaperHub

hub = PaperHub(language="tr")
hub.run(period="month", year=2026, month=5, top_n=10)
```

Per-call language override:

```python
hub.run(period="day", year=2026, month=5, day=1, top_n=5, language="tr")
```

For a step-by-step notebook, open `examples/03_jupyter_quickstart.ipynb`
(English) or `examples/03_jupyter_quickstart_tr.ipynb` (Turkish).

## Provider Selection

```python
PaperHub(provider="openai")                             # OpenAI default model
PaperHub(provider="openai", model="gpt-4o-mini")
PaperHub(provider="anthropic")                          # Anthropic default model
PaperHub(model="claude-3-5-haiku-20241022", provider="anthropic")
PaperHub(model="gemini-2.5-pro", provider="google")
```

If `model` is omitted, PaperHub picks the selected provider's default model.
Provider-specific `.env` values such as `PAPERHUB_OPENAI_MODEL` override those
defaults. `PAPERHUB_MODEL` remains available as a global override.

Provider SDKs are imported lazily — installing `paperhub` includes OpenAI by
default, and does not require Anthropic or Google packages unless you use those
providers.

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
python3 -m pytest          # unit tests, no live network or LLM keys needed
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m mypy src tests
python3 -m build           # build wheel + sdist
```

The unit tests use a fake LLM client and `httpx.MockTransport`; no real
provider keys are required.

## Project Layout

```
src/paperhub/
  __init__.py           # PaperHub public API
  config.py             # Settings (pydantic-settings)
  models.py             # PaperMeta, PaperSummary, RunRequest
  dates.py              # period → (start, end)
  fetchers/             # HF JSON API (default) + HTML fallback
  pdf/                  # arXiv download + text extraction
  agents/               # LLMClient protocol + provider clients + PaperAgent
  orchestrator.py       # asyncio.Semaphore parallelism
  cache.py              # SQLite cache
  formatter.py          # Markdown / plain-text rendering
  interactive_cli.py    # `paperhub` interactive launcher
tests/                  # pytest suite, mocked HTTP and fake LLM
docs/ARCHITECTURE.md
docs/HOW_TO_START.md
docs/API_KEYS.md
examples/README.md
examples/03_jupyter_quickstart.ipynb
examples/03_jupyter_quickstart_tr.ipynb
```

## Troubleshooting

- *"No papers found"*: HuggingFace may not yet have published Daily Papers
  for that date. Use `/metadata` in the interactive launcher to check the
  fetcher without an LLM call.
- *PDF text comes back tiny*: some arXiv PDFs use unusual layouts. PaperHub
  falls back to `pdfplumber`; if both extractors are short, the agent will
  pass through the abstract as the input text instead of failing.
