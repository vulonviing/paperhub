"""PaperHub quickstart example.

Run from the repository root:

    python examples/01_quickstart.py

Requires `ANTHROPIC_API_KEY` (or `--no-summarize` via the CLI for a
metadata-only sanity run).
"""

from __future__ import annotations

import os
import sys

from paperhub import PaperHub


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set. Either copy .env.example to .env and fill it in, "
            "or run `paperhub --no-summarize 'may 2026 top 5'` for a metadata-only check.",
            file=sys.stderr,
        )
        return 3

    hub = PaperHub(model="claude-3-5-haiku-20241022", concurrency=5)
    summaries = hub.run("may 2026 top 5 papers", display=False)
    for s in summaries:
        if s.error:
            print(f"[error] {s.title}: {s.error}")
            continue
        print(f"- {s.title} ({len(s.summary)} char) - {s.real_world_examples[:1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
