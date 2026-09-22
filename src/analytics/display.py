from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.models import RAGResult

console = Console()


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def show_result(result: RAGResult, show_scores: bool = True):
    verdict = result.verdict

    if verdict.in_pdf:
        banner = "[bold green]✔ ANSWERED FROM THE PDF[/]"
        border = "green"
    else:
        banner = "[bold yellow]⚠  NOT FROM THE PDF — answered from the model's general knowledge[/]"
        border = "yellow"

    console.print()
    console.print(Panel(Markdown(result.answer or "_(empty answer)_"), title=banner, border_style=border))

    if verdict.in_pdf and result.pages:
        sources = sorted({rc.chunk.source for rc in result.chunks})
        console.print(f"[dim]Sources:[/] {', '.join(sources)} — pages {', '.join(map(str, result.pages))}")

    if show_scores:
        show_analytics(result)


def show_analytics(result: RAGResult):
    verdict = result.verdict
    console.print(
        f"[dim]Verdict:[/] {'IN PDF' if verdict.in_pdf else 'OUT OF PDF'}  "
        f"[dim]method:[/] {verdict.method}  [dim]top relevance:[/] {verdict.top_score:.3f}"
    )
    console.print(f"[dim]{verdict.reason}[/]")

    if result.chunks:
        table = Table(title="Retrieved chunks", title_justify="left", show_lines=False)
        table.add_column("#", justify="right")
        table.add_column("Page", justify="right")
        table.add_column("Rerank", justify="right", style="bold")
        table.add_column("Dense (rank)", justify="right")
        table.add_column("BM25 (rank)", justify="right")
        table.add_column("RRF", justify="right")
        table.add_column("Preview", overflow="ellipsis", no_wrap=True, max_width=50)

        for i, rc in enumerate(result.chunks, start=1):
            dense = f"{_fmt(rc.dense_score)} ({rc.dense_rank})" if rc.dense_rank else "-"
            bm25 = f"{_fmt(rc.bm25_score, 2)} ({rc.bm25_rank})" if rc.bm25_rank else "-"
            table.add_row(
                str(i), str(rc.chunk.page), _fmt(rc.rerank_score), dense, bm25,
                _fmt(rc.rrf_score, 4), rc.chunk.text[:120],
            )
        console.print(table)

    t = result.timings_ms
    console.print(
        "[dim]Latency:[/] "
        + "  ".join(f"{name.removesuffix('_ms')} {t[name]:.0f}ms" for name in
                    ("dense_ms", "bm25_ms", "rerank_ms", "scope_ms", "llm_ms", "total_ms") if name in t)
    )


def show_sources(result: RAGResult):
    if not result.verdict.in_pdf:
        console.print("[yellow]These were the closest chunks, but they were judged irrelevant and NOT used for the answer.[/]")
    for i, rc in enumerate(result.chunks, start=1):
        console.print(Panel(rc.chunk.text, title=f"[{i}] {rc.chunk.source} — page {rc.chunk.page}", border_style="dim"))
