# PaperHub Examples

The `examples/` folder contains notebooks and scripts for trying PaperHub.

## Terminal CLI Launcher (Primary)

Start the interactive launcher from any terminal:

```bash
paperhub
```

Inside the launcher, type `/help` to see the command list. Common commands:

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
/api-keys
/clear-cache
/clear-cache summaries
/clear-cache pdfs
/clear-cache keys openai
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

The launcher shows the current run details — provider, model, API key status,
date range, top_n — in a dashboard. Use `/set-key` to save provider keys in
PaperHub's user config file, and use `/provider` and `/model` to switch
between providers and models, including local Ollama models. `/metadata` is
safe for setup checks because it does not call an LLM. `/run` uses the
selected provider; cloud providers require an API key, while Ollama requires a
running local Ollama server and a pulled model.

## Notebook Quickstarts

- `03_jupyter_quickstart.ipynb`: notebook quickstart.
