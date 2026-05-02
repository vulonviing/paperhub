# PaperHub API Keys

PaperHub only needs an API key for the provider you choose for summarization.
Metadata-only checks do not require any LLM key.

## Environment Variables

| Provider | Environment variable | Extra install |
|----------|----------------------|---------------|
| OpenAI | `OPENAI_API_KEY` | included by default |
| Anthropic | `ANTHROPIC_API_KEY` | `python3 -m pip install -e ".[anthropic]"` |
| Google Gemini | `GOOGLE_API_KEY` | `python3 -m pip install -e ".[google]"` |

## Recommended `.env` Setup

From the repository root:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

PAPERHUB_PROVIDER=openai
PAPERHUB_CONCURRENCY=2
PAPERHUB_MAX_PDF_CHARS=60000
```

Do not commit `.env`; the repository `.gitignore` excludes it.

## Provider Selection

Use the launcher:

```text
/provider
/provider openai
/provider anthropic
/provider google
```

## Missing Key Behavior

If the selected provider key is missing, PaperHub will stop before making an
LLM call and print setup guidance. You can still check the fetch pipeline:

```text
/metadata
```

In the interactive launcher, use:

```text
/status
/provider
/api-keys
/metadata
```

`/status` shows the selected provider key state. `/provider` opens the full
provider/API-key availability selector. `/api-keys` prints this guide in the
terminal.
