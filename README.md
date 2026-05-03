# PaperHub

PaperHub fetches HuggingFace Daily Papers by programmatic date filters, assigns
each paper to its own AI summarization agent, and renders Jupyter-friendly
summaries by default.

Two ways to use PaperHub:

1. **Terminal CLI** (`paperhub`) — interactive launcher with a REPL interface. **Primary mode.**
2. **Python / Jupyter** — import `PaperHub` and call `hub.run(...)` directly.

## Install

```bash
pip install paperhub
```

Optional provider extras:

```bash
pip install "paperhub[anthropic]"  # adds the Anthropic client
pip install "paperhub[google]"     # adds the Google Gemini client
```

For local models via Ollama, no extra package is needed — install
[Ollama](https://ollama.com) and pull the model before use:

```bash
ollama pull gemma4:e2b   # ~2 B parameters, fast on CPU
ollama pull gemma4:e4b   # ~4 B parameters, higher quality
```

## Configuration

PaperHub reads configuration from shell environment variables and from its own
per-user config file. The CLI can save provider keys for you without touching a
project-level `.env` file:

```bash
paperhub set-key openai
paperhub set-key anthropic
paperhub set-key google
paperhub check-llm
paperhub check-llm ollama     # no key needed — checks local Ollama server
paperhub api-keys
paperhub config-path
```

Inside the interactive launcher, use `/set-key openai`. The config file lives
under the OS-specific user config directory, for example
`~/Library/Application Support/paperhub/.env` on macOS. Environment variables
still override values saved there.

After `set-key`, PaperHub immediately sends one tiny request to the selected
provider/model and reports whether the key and LLM are working. You can repeat
that check later with `paperhub check-llm` or `/check-llm`.

| Variable                           | Purpose                                                         |
|------------------------------------|-----------------------------------------------------------------|
| `OPENAI_API_KEY`                   | Default provider key                                            |
| `ANTHROPIC_API_KEY`                | Optional, used when `--provider anthropic`                      |
| `GOOGLE_API_KEY`                   | Optional, used when `--provider google`                         |
| `PAPERHUB_PROVIDER`                | Override default provider (default: openai)                     |
| `PAPERHUB_MODEL`                   | Optional global model override                                  |
| `PAPERHUB_OPENAI_MODEL`            | OpenAI default model (default: gpt-5.4-mini)                    |
| `PAPERHUB_OPENAI_REASONING_EFFORT` | OpenAI reasoning effort (default: medium)                       |
| `PAPERHUB_ANTHROPIC_MODEL`         | Anthropic default model (default: claude-sonnet-4-6)            |
| `PAPERHUB_GOOGLE_MODEL`            | Google default model (default: gemini-3-flash-preview)          |
| `PAPERHUB_OLLAMA_MODEL`            | Ollama default model (default: gemma4:e2b)                      |
| `PAPERHUB_OLLAMA_BASE_URL`         | Ollama server URL (default: http://localhost:11434/v1)           |
| `PAPERHUB_CONCURRENCY`             | Max concurrent paper agents (default: 5)                        |
| `PAPERHUB_MAX_PDF_CHARS`           | Truncation cap for PDF text (default: 60000)                    |
| `PAPERHUB_CACHE_DIR`               | Override the on-disk cache location                             |
| `PAPERHUB_REQUEST_TIMEOUT_S`       | HTTP request timeout in seconds (default: 30)                   |

## Terminal CLI (Primary)

Start the interactive launcher:

```bash
paperhub
```

The launcher opens a REPL with a status dashboard showing the current provider,
model, API key state, date range, and top paper count. Use commands to configure
and run:

```text
/help
/status
/version
/provider
/provider openai
/provider ollama
/model
/model gpt-5.4-mini
/model gpt-4.1-mini
/model default
/language
/language tr
/date 2026-05
/date 2026-05-15
/date 2026-W18
/date 2026-05-01 2026-05-31
/top 5
/concurrency 2
/metadata
/run
/set-key openai
/keys
/check-llm
/check-llm ollama
/config-path
/guide
/api-keys
/clear-cache
/clear-cache summaries
/clear-cache pdfs
/clear-cache keys openai
/clear
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
If the selected provider key is missing, `/run` prints setup guidance.
`/api-keys` shows key status and setup help. `/check-llm` sends a tiny live
provider request and confirms that the selected key/model can respond.
`/clear-cache` deletes cached summaries/PDFs, and `/clear-cache keys openai`
removes a saved cloud-provider key from PaperHub's user config.

CLI startup commands are available without entering the launcher:

```bash
paperhub version
paperhub set-key openai
paperhub keys
paperhub check-llm
paperhub check-llm ollama
paperhub api-keys
paperhub config-path
paperhub clear-cache summaries
paperhub clear-cache keys openai
```

You can also pass startup flags:

```bash
paperhub --provider anthropic
paperhub --model claude-sonnet-4-6
paperhub --provider ollama --model gemma4:e2b
paperhub --language tr
paperhub --top-n 10
paperhub --concurrency 2
```

## Python / Jupyter API

```python
from datetime import date
from paperhub import PaperHub

# OpenAI — default reasoning model
hub = PaperHub(provider="openai")
hub.run(period="month", year=2026, month=5, top_n=10)

# OpenAI — budget option (standard chat pricing, no reasoning tokens)
hub = PaperHub(provider="openai", model="gpt-4.1-mini")
hub.run(period="month", year=2026, month=5, top_n=10)

# Local model via Ollama (no API key required)
hub = PaperHub(provider="ollama", model="gemma4:e2b")
hub.run(period="month", year=2026, month=5, top_n=5)

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

For a step-by-step notebook, open `examples/03_jupyter_quickstart.ipynb`.

## Provider Selection

```python
PaperHub(provider="openai")                             # OpenAI default model
PaperHub(provider="openai", model="gpt-5.4-mini")       # reasoning model (default)
PaperHub(provider="openai", model="gpt-4.1-mini")       # budget: no reasoning tokens
PaperHub(provider="anthropic")                          # Anthropic default model
PaperHub(provider="anthropic", model="claude-sonnet-4-6")
PaperHub(provider="google", model="gemini-3-flash-preview")
PaperHub(provider="ollama", model="gemma4:e2b")         # local — no API key needed
PaperHub(provider="ollama", model="gemma4:e4b")         # local — higher quality
```

If `model` is omitted, PaperHub picks the selected provider's default model.
Provider-specific config values such as `PAPERHUB_OPENAI_MODEL` override those
defaults. `PAPERHUB_MODEL` remains available as a global override. OpenAI uses
`PAPERHUB_OPENAI_REASONING_EFFORT=medium` by default; set it to an empty value
to let the OpenAI API choose its model default.

Provider SDKs are imported lazily — installing `paperhub` includes the OpenAI
SDK by default. Anthropic and Google require their optional extras. Ollama uses
its native local HTTP API first and falls back to Ollama's OpenAI-compatible
endpoint, so no extra Python package is needed beyond the base install.

### OpenAI model guide

| Model | Type | When to use |
|---|---|---|
| `gpt-5.4-mini` (default) | Reasoning | Best quality, uses reasoning tokens |
| `gpt-4.1-mini` | Standard chat | Budget option — lower cost, no reasoning budget |

### Local models via Ollama

PaperHub connects to Ollama's OpenAI-compatible endpoint at
`http://localhost:11434/v1` by default. No API key is required.

```bash
# Install Ollama: https://ollama.com
ollama pull gemma4:e2b    # ~2 B params — good for CPU inference
ollama pull gemma4:e4b    # ~4 B params — better quality

# Verify before running PaperHub:
paperhub check-llm ollama
```

Override the server URL if Ollama runs on a different host:

```bash
export PAPERHUB_OLLAMA_BASE_URL=http://192.168.1.10:11434/v1
```

## Caching

PaperHub caches metadata, PDF text, and summaries in
the OS-specific user cache directory, for example
`~/Library/Caches/paperhub/paperhub.sqlite` on macOS and
`~/.cache/paperhub/paperhub.sqlite` on Linux. Override with
`PAPERHUB_CACHE_DIR`.
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
    openai_client.py    # OpenAI (default, included)
    anthropic_client.py # Anthropic (optional extra)
    google_client.py    # Google Gemini (optional extra)
    ollama_client.py    # Ollama local models (no extra needed)
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
```

## Troubleshooting

- *"No papers found"*: HuggingFace may not yet have published Daily Papers
  for that date. Use `/metadata` in the interactive launcher to check the
  fetcher without an LLM call.
- *PDF text comes back tiny*: some arXiv PDFs use unusual layouts. PaperHub
  falls back to `pdfplumber`; if both extractors are short, the agent will
  pass through the abstract as the input text instead of failing.
- *Ollama connection error*: make sure Ollama is running (`ollama serve`) and
  the model is pulled (`ollama pull gemma4:e2b`). Run `paperhub check-llm ollama`
  to verify before starting a full run.
