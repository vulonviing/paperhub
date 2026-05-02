# PaperHub Examples

The `examples/` folder contains notebooks and scripts for trying PaperHub.

## Terminal CLI Launcher (Primary)

Start the interactive launcher from any terminal:

```bash
paperhub
```

Inside the launcher, type `/help` to see the command list. Common commands:

```text
/provider
/provider openai
/model
/model gpt-4o-mini
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

The launcher shows the current run details — provider, model, API key status,
date range, top_n — in a dashboard. Use `/provider` and `/model` to switch
between providers and models. `/metadata` is safe for setup checks because it
does not call an LLM. `/run` uses the selected provider and requires that
provider's API key.

## Notebook Quickstarts

- `03_jupyter_quickstart.ipynb`: English notebook quickstart.
- `03_jupyter_quickstart_tr.ipynb`: Turkish notebook quickstart.
