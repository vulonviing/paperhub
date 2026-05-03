# PaperHub How To Start

PaperHub has two modes: the interactive terminal launcher (`paperhub`) and
the Python/Jupyter API. The terminal launcher is the primary mode.

## 1. Install

### From PyPI (recommended)

```bash
pip install paperhub
```

Install optional provider extras if you need Anthropic or Google:

```bash
pip install "paperhub[anthropic]"
pip install "paperhub[google]"
```

### From the repository (development)

```bash
git clone https://github.com/vulonviing/paperhub.git
cd paperhub
pip install -e ".[dev]"
```

## 2. Add API Keys

For cloud providers, save keys into PaperHub's per-user config:

```bash
paperhub set-key openai
paperhub check-llm
```

Add optional providers when needed:

```bash
paperhub set-key anthropic
paperhub set-key google
```

### Local models via Ollama (no API key needed)

Install [Ollama](https://ollama.com), start the server, and pull a model:

```bash
ollama pull gemma4:e2b   # ~2 B params — fast on CPU
ollama pull gemma4:e4b   # ~4 B params — better quality
paperhub check-llm ollama
```

PaperHub defaults to `openai`, but you can choose another provider with the
launcher's `/provider` command. Environment variables such as
`OPENAI_API_KEY` still work and take priority over saved user config values.
After each `set-key`, PaperHub immediately sends one tiny provider request and
prints whether the selected LLM accepted the key and returned JSON.

## 3. Start The Interactive Launcher

```bash
paperhub
```

The launcher opens a status dashboard and drops into a REPL where you can
configure and run the pipeline without leaving the terminal.

## 4. Common Launcher Commands

Inside the launcher:

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

### Date formats

| Command                           | Period  | Description          |
|----------------------------------|---------|----------------------|
| `/date 2026-05`                  | month   | May 2026             |
| `/date 2026`                     | year    | Full year 2026       |
| `/date 2026-05-15`               | day     | Single day           |
| `/date 2026-W18`                 | week    | ISO week 18 of 2026  |
| `/date 2026-05-01 2026-05-31`    | custom  | Inclusive range      |

`/provider` opens the provider/API-key availability selector. `/set-key`
saves a provider key to PaperHub's user config file and verifies it with a
tiny LLM request. `/check-llm` repeats that live provider check later — also
works with `/check-llm ollama` for the local provider. `/model` opens the
model selector with built-in models (including local Ollama models) plus a
custom model id option. `/metadata` fetches HuggingFace paper metadata only
and does not call an LLM. `/run` runs the full PDF and LLM summarization
pipeline. `/concurrency` controls how many paper agents run at once.
`/clear-cache` deletes cached summaries/PDFs, while `/clear-cache keys openai`
removes a saved cloud-provider API key.

Startup commands work directly from the shell:

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

## 5. Python / Jupyter API

```python
from datetime import date
from paperhub import PaperHub

# Cloud providers
hub = PaperHub(provider="openai", language="en")
summaries = hub.run(period="month", year=2026, month=5, top_n=5)

# Budget OpenAI option (standard chat model, no reasoning tokens)
hub = PaperHub(provider="openai", model="gpt-4.1-mini")
summaries = hub.run(period="month", year=2026, month=5, top_n=5)

# Local model via Ollama (no API key required)
hub = PaperHub(provider="ollama", model="gemma4:e2b")
summaries = hub.run(period="month", year=2026, month=5, top_n=5)

# Other period formats
summaries = hub.run(period="day", year=2026, month=5, day=1, top_n=5)
summaries = hub.run(period="week", year=2026, week=18, top_n=5)
summaries = hub.run(period="year", year=2026, top_n=20)
summaries = hub.run(
    period="custom",
    start=date(2026, 4, 15),
    end=date(2026, 4, 30),
    top_n=5,
)
```

For Turkish output:

```python
hub = PaperHub(language="tr")
summaries = hub.run(period="month", year=2026, month=5, top_n=5)
```
