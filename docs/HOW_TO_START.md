# PaperHub How To Start

PaperHub has two modes: the interactive terminal launcher (`paperhub`) and
the Python/Jupyter API. The terminal launcher is the primary mode.

## 1. Install From The Repository

From the repository root:

```bash
python3 -m pip install -e ".[dev]"
```

Install optional provider extras when you want to use Anthropic or Google:

```bash
python3 -m pip install -e ".[anthropic]"
python3 -m pip install -e ".[google]"
```

## 2. Add API Keys

Copy the environment template:

```bash
cp .env.example .env
```

Then fill at least one provider key:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

PaperHub defaults to `openai`, but you can choose another provider with the
launcher's `/provider` command.

## 3. Start The Interactive Launcher

```bash
paperhub
```

The launcher opens a status dashboard and drops into a REPL where you can
configure and run the pipeline without leaving the terminal.

## 4. Common Launcher Commands

Inside the launcher:

```text
/status
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

### Date formats

| Command                           | Period  | Description          |
|----------------------------------|---------|----------------------|
| `/date 2026-05`                  | month   | May 2026             |
| `/date 2026`                     | year    | Full year 2026       |
| `/date 2026-05-15`               | day     | Single day           |
| `/date 2026-W18`                 | week    | ISO week 18 of 2026  |
| `/date 2026-05-01 2026-05-31`    | custom  | Inclusive range      |

`/provider` opens the provider/API-key availability selector. `/model` opens
the model selector with three built-in models plus a custom model id option.
`/metadata` fetches HuggingFace paper metadata only and does not call an LLM.
`/run` runs the full PDF and LLM summarization pipeline.

## 5. Python / Jupyter API

```python
from datetime import date
from paperhub import PaperHub

hub = PaperHub(provider="openai", language="en")

summaries = hub.run(period="month", year=2026, month=5, top_n=5)
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
