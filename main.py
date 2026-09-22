from collections import Counter
from pathlib import Path
from typing import Optional

import typer

from config import settings
from src.analytics.display import console, show_analytics, show_result, show_sources

app = typer.Typer(add_completion=False, help="PDF RAG CLI — hybrid retrieval with Groq.")


def build_pipeline():
    from src.pipeline.rag_pipeline import RAGPipeline
    with console.status("Loading index..."):
        return RAGPipeline()


# Shown once a node FINISHES, so each label describes what happens next.
NEXT_STEP_LABELS = {
    "retrieve": "Checking whether the PDF covers it...",
    "check_scope": "Working on the answer...",
    "rewrite_query": "Weak match, searching again with a rewritten query...",
}


def run_question(pipeline, logger, question: str, show_scores: bool):
    with console.status("Searching the PDF...") as status:
        result = pipeline.answer(question, on_step=lambda node: status.update(NEXT_STEP_LABELS.get(node, "Working...")))
    logger.log(result, pipeline.groq_client.last_usage)
    show_result(result, show_scores=show_scores)
    return result


@app.command()
def ingest(path: Optional[Path] = typer.Argument(None, help="A PDF file or a folder of PDFs. Defaults to data/pdfs.")):
    """Extract, chunk, embed and index PDFs."""
    path = path or settings.pdf_dir
    pdfs = sorted(path.glob("*.pdf")) if path.is_dir() else [path]
    if not pdfs or not all(p.exists() for p in pdfs):
        console.print(f"[red]No PDF found at {path}[/]")
        raise typer.Exit(1)

    pipeline = build_pipeline()
    for pdf in pdfs:
        with console.status(f"Ingesting {pdf.name} (first run downloads the embedding model)..."):
            stats = pipeline.ingest(str(pdf))
        console.print(
            f"[green]✔[/] {stats['source']}: {stats['pages']} pages "
            f"({stats['empty_pages']} without text), {stats['chunks']} chunks in {stats['seconds']:.1f}s"
        )
    console.print(f"Index now holds {pipeline.vector_store.count()} chunks.")


@app.command()
def ask(question: str, scores: bool = typer.Option(True, help="Show the analytics panel.")):
    """Answer a single question."""
    from src.analytics.query_logger import QueryLogger
    pipeline = build_pipeline()
    if pipeline.vector_store.count() == 0:
        console.print("[yellow]The index is empty — run `python main.py ingest` first.[/]")
        raise typer.Exit(1)
    run_question(pipeline, QueryLogger(), question, scores)


@app.command()
def chat():
    """Interactive question loop."""
    from src.analytics.query_logger import QueryLogger
    pipeline = build_pipeline()
    if pipeline.vector_store.count() == 0:
        console.print("[yellow]The index is empty — run `python main.py ingest` first.[/]")
        raise typer.Exit(1)

    with console.status("Loading models..."):
        pipeline.retriever.retrieve("warm up")  # so the first question isn't slowed by model loading

    logger = QueryLogger()
    show_scores = True
    last = None
    console.print("[bold]PDF chat[/] — ask anything. Commands: /scores (toggle analytics), /sources, /quit")

    while True:
        try:
            question = console.input("\n[bold cyan]You ›[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            continue
        if question in ("/quit", "/exit"):
            break
        if question == "/scores":
            show_scores = not show_scores
            console.print(f"Analytics panel {'on' if show_scores else 'off'}.")
            if show_scores and last:
                show_analytics(last)
            continue
        if question == "/sources":
            if last and last.chunks:
                show_sources(last)
            else:
                console.print("No sources yet.")
            continue

        try:
            last = run_question(pipeline, logger, question, show_scores)
        except Exception as exc:
            console.print(f"[red]Error:[/] {exc}")


@app.command()
def stats():
    """Show index contents and the query-log summary."""
    from src.analytics.query_logger import QueryLogger
    from src.indexing.vector_store import VectorStore

    chunks = VectorStore().all_chunks()
    per_source = Counter(c.source for c in chunks)
    console.print(f"[bold]Index:[/] {len(chunks)} chunks from {len(per_source)} document(s)")
    for source, count in per_source.most_common():
        console.print(f"  • {source}: {count} chunks")

    records = QueryLogger().read_all()
    if not records:
        console.print("[bold]Queries:[/] none logged yet")
        return
    in_pdf = sum(r["in_pdf"] for r in records)
    methods = Counter(r["method"] for r in records)
    avg_ms = sum(r["timings_ms"].get("total_ms", 0) for r in records) / len(records)
    console.print(
        f"[bold]Queries:[/] {len(records)} total — {in_pdf} in PDF, {len(records) - in_pdf} out of PDF, "
        f"avg {avg_ms:.0f}ms"
    )
    console.print("  decided by: " + ", ".join(f"{m} {n}" for m, n in methods.most_common()))


@app.command()
def graph():
    """Print the LangGraph workflow as a Mermaid diagram."""
    from src.pipeline.rag_pipeline import RAGPipeline
    print(RAGPipeline().graph_mermaid())


@app.command()
def reset(yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt.")):
    """Delete the vector store and BM25 index."""
    if not yes and not typer.confirm("Delete all indexed chunks?"):
        raise typer.Abort()
    from src.pipeline.rag_pipeline import RAGPipeline
    RAGPipeline().reset()
    console.print("Index cleared.")


if __name__ == "__main__":
    app()
