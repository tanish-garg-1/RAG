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


def show_vision_summary(vision: dict, images_requested: bool):
    if not images_requested:
        return
    if not vision.get("enabled"):
        console.print(f"  [yellow]Images skipped:[/] {vision.get('reason', 'vision disabled')}")
        return
    used = ", ".join(f"{name} {n}" for name, n in vision["providers"].items()) or "none"
    console.print(
        f"  Images: {vision['images']} found — {vision['described']} described, "
        f"{vision['decorative']} decorative, {vision['failed']} failed  [dim](used: {used}; cached: {vision['cache_hits']})[/]"
    )
    for event in dict.fromkeys(vision["events"]):   # de-duplicated, order kept
        console.print(f"  [yellow]•[/] {event}")
    if vision["failed"]:
        console.print(
            f"  [yellow]{vision['failed']} image(s) could not be read[/] (last error: {vision['last_error']}). "
            "Run ingest again later: images already described are cached and cost nothing."
        )


@app.command("vision-status")
def vision_status_command():
    """Show whether Gemini and Ollama are ready for image understanding."""
    from src.vision.describer import vision_status
    status = vision_status()

    gemini = status["gemini"]
    key = "[green]key set[/]" if gemini["configured"] else "[yellow]no GEMINI_API_KEY[/]"
    console.print(f"[bold]1. Gemini[/] ({gemini['model']}): {key}")

    ollama = status["ollama"]
    if not ollama["reachable"]:
        console.print(f"[bold]2. Ollama[/] ({ollama['model']}): [yellow]server not reachable at {ollama['host']}[/]")
        return
    if ollama["model_installed"]:
        console.print(f"[bold]2. Ollama[/] ({ollama['model']}): [green]ready[/]")
    else:
        installed = ", ".join(ollama["installed"]) or "none"
        console.print(f"[bold]2. Ollama[/]: [yellow]server is up but '{ollama['model']}' is not pulled[/] (installed: {installed})")
        console.print(f"   Run: ollama pull {ollama['model']}   or set OLLAMA_MODEL to one of the installed names.")


@app.command()
def ingest(
    path: Optional[Path] = typer.Argument(None, help="A PDF file or a folder of PDFs. Defaults to data/pdfs."),
    images: bool = typer.Option(True, "--images/--no-images", help="Describe figures and scanned pages with a vision model."),
):
    """Extract, chunk, embed and index PDFs."""
    path = path or settings.pdf_dir
    pdfs = sorted(path.glob("*.pdf")) if path.is_dir() else [path]
    if not pdfs or not all(p.exists() for p in pdfs):
        console.print(f"[red]No PDF found at {path}[/]")
        raise typer.Exit(1)

    pipeline = build_pipeline()
    for pdf in pdfs:
        with console.status(f"Ingesting {pdf.name} (first run downloads the embedding model)...") as status:
            stats = pipeline.ingest(str(pdf), describe_images=images, on_progress=status.update)
        console.print(
            f"[green]✔[/] {stats['source']}: {stats['pages']} pages "
            f"({stats['empty_pages']} without text), {stats['chunks']} chunks "
            f"({stats['image_chunks']} from images) in {stats['seconds']:.1f}s"
        )
        show_vision_summary(stats["vision"], images)
    console.print(f"Index now holds {pipeline.vector_store.count()} chunks.")


@app.command()
def ask(question: str, scores: bool = typer.Option(True, help="Show the analytics panel.")):
    """Answer a single question."""
    from src.analytics.query_logger import QueryLogger
    if not question.strip():
        console.print("[yellow]Question is empty.[/]")
        raise typer.Exit(1)
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
