"""`paperhub` CLI entrypoint.

The CLI is intentionally thin: it parses the user query, drives `PaperHub`,
and prints plain text. If the LLM provider needs an API key that is missing,
we surface a clear message instead of a crashing stacktrace.
"""

from __future__ import annotations

import sys

import click

from . import PaperHub, parse_query
from .config import load_settings
from .formatter import render_plain
from .localization import normalize_language


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("query", nargs=-1)
@click.option(
    "--model",
    default=None,
    help="LLM model id (default: selected provider's configured default).",
)
@click.option(
    "--provider",
    default=None,
    help="Override provider: anthropic, openai, google.",
)
@click.option(
    "--top-n",
    type=int,
    default=None,
    help="Override the top_n parsed from the query.",
)
@click.option(
    "--concurrency",
    type=int,
    default=None,
    help="Max concurrent paper agents (default: 5).",
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    default=None,
    help="Override cache directory.",
)
@click.option(
    "--no-summarize",
    is_flag=True,
    default=False,
    help="Fetch papers and print metadata only (no LLM calls).",
)
@click.option(
    "--language",
    type=click.Choice(["en", "tr"], case_sensitive=False),
    default="en",
    show_default=True,
    help="Output language: en or tr.",
)
def main(
    query: tuple[str, ...],
    model: str | None,
    provider: str | None,
    top_n: int | None,
    concurrency: int | None,
    cache_dir: str | None,
    no_summarize: bool,
    language: str,
) -> None:
    """Run PaperHub from the terminal.

    Example:

        paperhub "may 2026 top 10 papers"
    """

    output_language = normalize_language(language)
    if not query:
        if output_language == "tr":
            message = (
                'Kullanım: paperhub --language tr "mayis 2026 top 10 paper"\n'
                "Türkçe veya İngilizce bir tarih ifadesi ver, örn. 'bugün top 5'."
            )
        else:
            message = (
                'Usage: paperhub "may 2026 top 10 papers"\n'
                "Provide a Turkish or English date phrase, e.g. 'today top 5'."
            )
        click.echo(message, err=True)
        sys.exit(2)

    text = " ".join(query).strip()
    settings = load_settings()

    try:
        request = parse_query(text)
    except Exception as exc:
        if output_language == "tr":
            click.echo(f"Sorgu ayrıştırılamadı: {exc}", err=True)
        else:
            click.echo(f"Could not parse query: {exc}", err=True)
        sys.exit(2)
    if top_n is not None:
        request = request.model_copy(update={"top_n": top_n})
    request = request.model_copy(update={"language": output_language})

    if no_summarize:
        _print_metadata_only(request, settings, cache_dir, concurrency)
        return

    if not _has_provider_key(provider or settings.paperhub_provider, settings):
        click.echo(
            _provider_help(provider or settings.paperhub_provider, output_language), err=True
        )
        sys.exit(3)

    hub = PaperHub(
        model=model,
        provider=provider,
        cache_dir=cache_dir,
        concurrency=concurrency,
        settings=settings,
        language=output_language,
    )
    summaries = hub.run(query=text, top_n=top_n, language=output_language, display=False)
    click.echo(render_plain(summaries, request))


def _has_provider_key(provider: str, settings) -> bool:
    provider = (provider or "anthropic").lower()
    return bool(
        {
            "anthropic": settings.anthropic_api_key,
            "openai": settings.openai_api_key,
            "google": settings.google_api_key,
        }.get(provider)
    )


def _provider_help(provider: str, language: str = "en") -> str:
    provider = (provider or "anthropic").lower()
    env_name = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }.get(provider, "ANTHROPIC_API_KEY")
    if normalize_language(language) == "tr":
        return (
            f"'{provider}' sağlayıcısı için API anahtarı eksik. ${env_name} ayarla "
            f"veya .env.example dosyasını .env olarak kopyalayıp doldur.\n"
            f"LLM çağrısı yapmadan metadata çekmek için --no-summarize kullan."
        )
    return (
        f"Missing API key for provider '{provider}'. Set ${env_name} or "
        f"copy .env.example to .env and fill it in.\n"
        f"Use --no-summarize to fetch metadata without an LLM call."
    )


def _print_metadata_only(
    request,
    settings,
    cache_dir: str | None,
    concurrency: int | None,
) -> None:
    """Fast path: fetch and print paper metadata without invoking an LLM."""

    import asyncio

    import httpx

    from .dates import resolve_range
    from .fetchers import HFAPIFetcher, HFHtmlFetcher, papers_in_range

    async def _fetch() -> list:
        start_date, end_date = resolve_range(request)
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=settings.paperhub_request_timeout_s,
        ) as client:
            primary = HFAPIFetcher(client)
            fallback = HFHtmlFetcher(client)
            return await papers_in_range(
                primary,
                start_date,
                end_date,
                request.top_n,
                fallback=fallback,
            )

    try:
        papers = asyncio.run(_fetch())
    except Exception as exc:
        if request.language == "tr":
            click.echo(f"Veri çekme başarısız: {exc}", err=True)
        else:
            click.echo(f"Fetch failed: {exc}", err=True)
        sys.exit(4)

    if not papers:
        if request.language == "tr":
            click.echo("Bu dönem için makale bulunamadı.")
        else:
            click.echo("No papers found for that period.")
        return

    if request.language == "tr":
        click.echo(f"{len(papers)} makale bulundu:")
    else:
        click.echo(f"Found {len(papers)} papers:")
    for idx, paper in enumerate(papers, 1):
        click.echo(f"{idx}. ▲{paper.upvotes} {paper.title}  [{paper.arxiv_id}]")


if __name__ == "__main__":  # pragma: no cover
    main()
