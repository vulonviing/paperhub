# PaperHub API Keys

PaperHub only needs an API key for the provider you choose for summarization.
Metadata-only checks do not require any LLM key.

## Recommended Setup

Use the CLI to save keys into PaperHub's own per-user config file:

```bash
paperhub version
paperhub set-key openai
paperhub set-key anthropic
paperhub set-key google
paperhub keys
paperhub check-llm
paperhub api-keys
paperhub config-path
```

Inside the interactive launcher, use:

```text
/set-key openai
/set-key anthropic
/set-key google
/check-llm
/api-keys
```

PaperHub stores these values in an app-specific dotenv file, not in the
current project's `.env`. On macOS this is typically
`~/Library/Application Support/paperhub/.env`; on Linux it is usually
`~/.config/paperhub/.env`. The file is written with user-only permissions
where the OS supports that.

Every `set-key` command saves the key and then sends one tiny request to the
selected provider/model. If the provider rejects the key, the model name is
wrong, an optional SDK is missing, or the LLM returns an empty response,
PaperHub reports that immediately.

## Environment Variables

Shell environment variables still work and take priority over the saved user
config file.

| Provider | Environment variable | Extra install |
|----------|----------------------|---------------|
| OpenAI | `OPENAI_API_KEY` | included by default |
| Anthropic | `ANTHROPIC_API_KEY` | `python3 -m pip install "paperhub[anthropic]"` |
| Google Gemini | `GOOGLE_API_KEY` | `python3 -m pip install "paperhub[google]"` |

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
/set-key openai
/check-llm
/api-keys
/metadata
```

`/status` shows the selected provider key state. `/provider` opens the full
provider/API-key availability selector. `/check-llm` repeats the live provider
check without changing saved keys. `/api-keys` prints this guide in the
terminal.
