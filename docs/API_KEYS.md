# PaperHub API Keys

PaperHub only needs an API key for the cloud provider you choose for
summarization. Local Ollama models and metadata-only checks do not require
any key.

## Recommended Setup

Use the CLI to save keys into PaperHub's own per-user config file:

```bash
paperhub set-key openai
paperhub set-key anthropic
paperhub set-key google
paperhub keys
paperhub check-llm
paperhub check-llm ollama
paperhub api-keys
paperhub config-path
paperhub clear-cache keys openai
```

Inside the interactive launcher, use:

```text
/set-key openai
/set-key anthropic
/set-key google
/check-llm
/check-llm ollama
/api-keys
/clear-cache keys openai
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

| Provider | Environment variable | Install |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | included by default |
| Anthropic | `ANTHROPIC_API_KEY` | `pip install "paperhub[anthropic]"` |
| Google Gemini | `GOOGLE_API_KEY` | `pip install "paperhub[google]"` |
| Ollama (local) | — | no key needed; install [Ollama](https://ollama.com) |

## Local Models via Ollama

Ollama runs models locally — no API key or network access required for
inference. Install Ollama, pull a model, then verify:

```bash
# Install: https://ollama.com
ollama pull gemma4:e2b   # ~2 B parameters — fast on most hardware
ollama pull gemma4:e4b   # ~4 B parameters — better quality

# Verify the connection before running PaperHub:
paperhub check-llm ollama
```

Inside the launcher:

```text
/provider ollama
/model gemma4:e2b
/check-llm ollama
/run
```

If Ollama runs on a different host, set:

```bash
export PAPERHUB_OLLAMA_BASE_URL=http://192.168.1.10:11434/v1
```

## Provider Selection

Use the launcher:

```text
/provider
/provider openai
/provider anthropic
/provider google
/provider ollama
```

## Missing Key Behavior

If the selected cloud provider key is missing, PaperHub will stop before
making an LLM call and print setup guidance. You can still check the fetch
pipeline:

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

To remove a saved cloud-provider key from PaperHub's user config, use:

```text
/clear-cache keys openai
/clear-cache keys anthropic
/clear-cache keys google
/clear-cache keys all
```
