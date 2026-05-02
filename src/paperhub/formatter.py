"""Render `PaperSummary` lists for Jupyter (Markdown) or terminals (plain text)."""

from __future__ import annotations

from collections.abc import Iterable

from .dates import pretty_period
from .localization import OutputLanguage, normalize_language
from .models import PaperSummary, RunRequest

_LABELS: dict[OutputLanguage, dict[str, str]] = {
    "en": {
        "paper_unit": "papers",
        "error": "Error",
        "no_examples": "- (no examples found)",
        "motivation": "Motivation",
        "method": "Method",
        "findings": "Findings",
        "examples": "Real-world examples",
        "summary": "Summary",
        "chars": "chars",
    },
    "tr": {
        "paper_unit": "makale",
        "error": "Hata",
        "no_examples": "- (örnek bulunamadı)",
        "motivation": "Motivasyon",
        "method": "Yöntem",
        "findings": "Bulgular",
        "examples": "Gerçek dünya örnekleri",
        "summary": "Özet",
        "chars": "kar.",
    },
}


def render_markdown(summaries: Iterable[PaperSummary], request: RunRequest) -> str:
    """Build the Markdown string used for Jupyter rendering and Markdown export."""

    summaries = list(summaries)
    labels = _LABELS[normalize_language(request.language)]
    header = f"# PaperHub - {pretty_period(request)} · {len(summaries)} {labels['paper_unit']}"
    blocks: list[str] = [header]

    for idx, s in enumerate(summaries, start=1):
        if s.error:
            blocks.append(f"## {idx}. {s.title}\n\n> {labels['error']}: {s.error}")
            continue
        examples = "\n".join(f"- {ex}" for ex in s.real_world_examples) or labels["no_examples"]
        blocks.append(
            "\n\n".join(
                [
                    f"## {idx}. {s.title}",
                    f"`arxiv:{s.arxiv_id}` · "
                    f"[PDF](https://arxiv.org/pdf/{s.arxiv_id}) · "
                    f"[HF](https://huggingface.co/papers/{s.arxiv_id})",
                    f"**{labels['motivation']}.** {s.motivation}",
                    f"**{labels['method']}.** {s.method}",
                    f"**{labels['findings']}.** {s.findings}",
                    f"**{labels['examples']}:**\n{examples}",
                    f"**{labels['summary']}:**\n\n{s.summary}",
                ]
            )
        )
    return "\n\n".join(blocks)


def render_plain(summaries: Iterable[PaperSummary], request: RunRequest) -> str:
    """Plain-text rendering for non-notebook environments."""

    summaries = list(summaries)
    labels = _LABELS[normalize_language(request.language)]
    lines: list[str] = []
    title = f"PaperHub - {pretty_period(request)} · {len(summaries)} {labels['paper_unit']}"
    lines.append(title)
    lines.append("=" * len(title))

    for idx, s in enumerate(summaries, start=1):
        lines.append("")
        lines.append(f"{idx}. {s.title}")
        lines.append(f"   arxiv:{s.arxiv_id}  PDF: https://arxiv.org/pdf/{s.arxiv_id}")
        if s.error:
            lines.append(f"   {labels['error']}: {s.error}")
            continue
        lines.append(f"   {labels['motivation']}: {s.motivation}")
        lines.append(f"   {labels['method']}: {s.method}")
        lines.append(f"   {labels['findings']}: {s.findings}")
        if s.real_world_examples:
            lines.append(f"   {labels['examples']}:")
            for ex in s.real_world_examples:
                lines.append(f"     - {ex}")
        lines.append(f"   {labels['summary']} ({len(s.summary)} {labels['chars']}): {s.summary}")
    return "\n".join(lines)


def display_summaries(
    summaries: Iterable[PaperSummary], request: RunRequest, *, force_plain: bool = False
) -> str:
    """Render summaries; in IPython call `display(Markdown(...))`.

    Always returns the Markdown string; the Jupyter render is a side effect.
    Falls back to plain text if IPython is unavailable or `force_plain=True`.
    """

    md = render_markdown(summaries, request)
    if force_plain:
        return md
    try:
        from IPython.display import Markdown, display

        if _running_in_ipython():
            display(Markdown(md))
    except Exception:
        pass
    return md


def _running_in_ipython() -> bool:
    try:
        from IPython import get_ipython
    except Exception:
        return False
    try:
        shell = get_ipython()
    except Exception:
        return False
    if shell is None:
        return False
    name = shell.__class__.__name__
    return name in {"ZMQInteractiveShell", "Shell", "TerminalInteractiveShell"}
