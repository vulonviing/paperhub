"""Programmatic period example.

Demonstrates programmatic period selection (no NL parsing) and a non-Anthropic
provider override. Requires the relevant API key for the chosen provider.
"""

from __future__ import annotations

from datetime import date

from paperhub import PaperHub


def main() -> None:
    hub = PaperHub(model="claude-3-5-haiku-20241022", concurrency=5)

    # Whole month
    hub.run(period="month", year=2026, month=5, top_n=10)

    # ISO week
    hub.run(period="week", year=2026, week=18, top_n=5)

    # Custom range
    hub.run(
        period="custom",
        start=date(2026, 4, 15),
        end=date(2026, 4, 30),
        top_n=15,
    )


if __name__ == "__main__":
    main()
