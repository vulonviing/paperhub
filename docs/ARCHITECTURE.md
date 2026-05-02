# PaperHub Architecture Notes

This document captures the engineering choices behind PaperHub and complements
the user-facing workflow in `README.md`.

## Components

```
            programmatic args (period, year, month, …)
                    │
                    ▼
              RunRequest
                    │
                    ▼
           dates.resolve_range  ──► (start, end)
                    │
                    ▼
         fetchers.papers_in_range
              ├─ HFAPIFetcher (Plan A)
              └─ HFHtmlFetcher (Plan B, fallback)
                    │
                    ▼  list[PaperMeta]
              orchestrator.run_all
                    │  Semaphore(5)
                    ▼
                 PaperAgent x N
              ├─ PDF download (cached)
              ├─ extract_text (pypdf → pdfplumber fallback)
              ├─ LLMClient.complete (anthropic | openai | google)
              └─ post-validation (<=6000 chars, JSON repair)
                    │
                    ▼  list[PaperSummary]
                formatter
              ├─ render_markdown (Jupyter)
              └─ render_plain (terminal launcher)
```

## Boundaries

- **HuggingFace access order is JSON API → HTML scraping**. The HTML fallback
  is only triggered when the JSON API errors out for a given day.
- **PDF text extraction** uses `pypdf` first (fast); if the result is too
  short it tries `pdfplumber`. Both extractors are wrapped to never raise.
- **LLM access** is gated by an `LLMClient` protocol. `PaperAgent` knows
  nothing about Anthropic/OpenAI/Google internals. Provider SDKs are imported
  lazily inside their respective client classes.
- **Cache** is a single SQLite database under the OS-specific user cache
  directory. Three
  tables: `paper_meta`, `pdf_text`, and `summary(arxiv_id, model)`. Summary
  cache keys include the model identifier so swapping models does not return
  stale results.

## Failure isolation

- Per-day fetch failures fall through to the HTML fallback for that day; if
  both fail the day is dropped from the aggregate (logged warning).
- Per-paper agent failures (PDF 404, extractor errors, LLM errors) are caught
  and recorded as `PaperSummary(error=...)` so a single failure never breaks
  the run.

## Output Language

PaperHub defaults to English output. The public entrypoints accept
`language="en"` or `language="tr"`:

- `PaperHub(language="tr")` sets the default for an instance.
- `hub.run(..., language="tr")` sets it for a single call.
- `/language` sets it in the interactive launcher.

The selected language is stored on `RunRequest` and passed through the
formatter, orchestrator, and `PaperAgent`. Prompt templates are selected in
`agents.prompts`, so the same JSON schema is returned in either language while
field values are generated in English or Turkish. Summary cache keys include
`(arxiv_id, model, language)` to prevent a cached Turkish summary from being
returned for an English request, or vice versa.

`PaperSummary.summary` is the canonical summary field. The old `summary_tr`
input/output shape is still accepted as a compatibility alias when loading old
payloads or user code constructs `PaperSummary(summary_tr=...)`.

## Length Enforcement

The 6000-character cap on `summary` is enforced in three places:

1. The system prompt instructs the model that the limit is hard in the selected
   output language.
2. `paper_agent._build_summary` calls `trim_to_sentence_boundary` on the
   model output before constructing the Pydantic model.
3. The `PaperSummary.summary` field has `max_length=6000` and a
   `field_validator` that double-checks at construction time.

If JSON parsing fails on the first attempt, the agent retries once with an
explicit "your previous output was invalid; produce JSON only" instruction.

## Concurrency

The orchestrator uses `asyncio.gather` with an `asyncio.Semaphore(5)` by
default. This is also a polite concurrency level for the arXiv PDF mirror.

## Configuration

`config.Settings` reads from environment variables and PaperHub's app-specific
user config dotenv file. The CLI writes that file with `paperhub set-key` or
interactive `/set-key`, instead of modifying the current directory's generic
`.env`. After saving a key, the CLI runs a tiny live provider health check via
`agents.health.check_llm` so bad keys, wrong model ids, missing optional SDKs,
and empty provider responses show up immediately. Provider clients still only
require keys when a remote call is made, which lets imports and the unit test
suite run without provider credentials.

Model resolution is provider-aware. If the caller omits `model`, `PaperHub`
uses the selected provider's default model (`PAPERHUB_ANTHROPIC_MODEL`,
`PAPERHUB_OPENAI_MODEL`, or `PAPERHUB_GOOGLE_MODEL`). `PAPERHUB_MODEL` is a
global override, but when it clearly belongs to a different provider it is
ignored so that a Claude model id is not sent to OpenAI or Google.
OpenAI also reads `PAPERHUB_OPENAI_REASONING_EFFORT`, defaulting to `xhigh`
for GPT-5.4 mini unless the value is set empty. OpenAI reasoning models use a
larger minimum completion budget because hidden reasoning tokens and visible
JSON output share the same completion limit.

## Why these tradeoffs

- Choosing OpenAI as the default keeps the base install aligned with the
  default interactive launcher workflow. Anthropic/Google are extras to avoid
  forcing users to install three SDKs.
- Hatchling as the build backend matches the PEP 621 manifest with no plugin
  configuration.
- SQLite avoids a service dependency and survives ~100k papers comfortably.
- Date input is strictly programmatic (period/year/month/…) — this eliminates
  a parsing layer and makes the API predictable for both code and the REPL.
