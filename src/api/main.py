import shutil
import tempfile
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile

from config import settings
from src.analytics.query_logger import QueryLogger
from src.api.schemas import ChatRequest, ChatResponse, Explanation, HealthResponse, IngestResponse, Source
from src.monitoring import metrics

MAX_UPLOAD_MB = 50


def create_app(pipeline=None, query_logger: QueryLogger | None = None, warm_up: bool = True) -> FastAPI:
    """Builds the API. Passing a `pipeline` (a fake in tests) skips loading the real models."""
    state = {"pipeline": pipeline}
    logger = query_logger or QueryLogger()
    lock = threading.Lock()   # the pipeline (Chroma, BM25, graph) is shared state: one request at a time

    init_lock = threading.Lock()   # the warm-up thread and the first request must not both build the pipeline (Chroma crashes)

    def get_pipeline():
        with init_lock:
            if state["pipeline"] is None:
                from src.pipeline.rag_pipeline import RAGPipeline
                state["pipeline"] = RAGPipeline()
            return state["pipeline"]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if warm_up and pipeline is None:
            def load():
                p = get_pipeline()
                if p.vector_store.count():
                    p.retriever.retrieve("warm up")   # load the embedding + reranker models now, not on the first user question
            threading.Thread(target=load, daemon=True).start()
        yield

    app = FastAPI(
        title="PDF RAG API",
        description="Hybrid-retrieval question answering over PDFs, with an explainable in-PDF / out-of-PDF decision.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health():
        p = get_pipeline()
        return HealthResponse(
            status="ok",
            chunks_indexed=p.vector_store.count(),
            gate_loaded=p.gate is not None,
            gate_version=p.gate.version if p.gate else None,
            groq_configured=bool(settings.groq_api_key),
            azure_configured=settings.azure_enabled,
        )

    @app.get("/documents")
    def documents():
        chunks = get_pipeline().vector_store.all_chunks()
        per_source = Counter(c.source for c in chunks)
        return {"total_chunks": len(chunks), "documents": dict(per_source)}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(file: UploadFile = File(...), images: bool = Query(False, description="Describe figures / scanned pages with Gemini.")):
        name = Path(file.filename or "").name
        if not name.lower().endswith(".pdf"):
            raise HTTPException(400, "Only .pdf files are accepted.")
        # The upload is staged in a temp folder and only copied into the PDF folder once ingest succeeds,
        # so a corrupt upload can neither leave junk behind nor overwrite a good file of the same name.
        with tempfile.TemporaryDirectory() as staging:
            staged = Path(staging) / name
            with open(staged, "wb") as out:
                shutil.copyfileobj(file.file, out)
            if staged.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")

            with lock:
                try:
                    stats = get_pipeline().ingest(str(staged), describe_images=images)
                except ValueError as exc:   # unreadable PDF, or a scanned PDF with no text and no vision provider
                    metrics.ERRORS.labels("ingest").inc()
                    raise HTTPException(422, str(exc))
            settings.pdf_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staged, settings.pdf_dir / name)
        metrics.INGESTS.inc()
        return IngestResponse(**{k: stats[k] for k in ("source", "pages", "chunks", "image_chunks", "seconds", "total_chunks")})

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest):
        p = get_pipeline()
        if p.vector_store.count() == 0:
            raise HTTPException(409, "The index is empty. Upload a PDF first (POST /ingest).")

        start = time.perf_counter()
        with lock:
            fallbacks_before = p.llm.stats["fallbacks"]
            try:
                result = p.answer(request.question, persona=request.persona)
            except RuntimeError as exc:   # every LLM provider failed
                metrics.ERRORS.labels("chat").inc()
                raise HTTPException(502, str(exc))
            explanation = p.explain(result) if request.explain else None
            usage = p.llm.last_usage
            fell_back = p.llm.stats["fallbacks"] > fallbacks_before

        logger.log(result, usage)
        metrics.QUESTIONS.labels("in_pdf" if result.verdict.in_pdf else "out_of_pdf").inc()
        metrics.LLM_PROVIDER.labels(result.provider or "unknown").inc()
        if fell_back:
            metrics.FALLBACKS.inc()
        metrics.LATENCY.observe(time.perf_counter() - start)

        return ChatResponse(
            answer=result.answer,
            in_pdf=result.verdict.in_pdf,
            verdict_method=result.verdict.method,
            verdict_reason=result.verdict.reason,
            gate_probability=result.verdict.gate_proba,
            top_score=result.verdict.top_score,
            provider=result.provider,
            persona=result.persona,
            path=result.path,
            rewritten_queries=result.rewritten_queries,
            sources=[
                Source(source=rc.chunk.source, page=rc.chunk.page, kind=rc.chunk.kind,
                       rerank_score=rc.rerank_score, text=rc.chunk.text)
                for rc in result.chunks
            ],
            timings_ms=result.timings_ms,
            explanation=Explanation(**explanation) if explanation else None,
        )

    @app.get("/stats")
    def stats():
        """Everything the dashboard shows: query-log summary, LLM provider counters, gate info."""
        p = get_pipeline()
        records = logger.read_all()
        totals = sorted(r["timings_ms"].get("total_ms", 0) for r in records)
        return {
            "queries": len(records),
            "in_pdf": sum(r["in_pdf"] for r in records),
            "out_of_pdf": sum(not r["in_pdf"] for r in records),
            "decided_by": dict(Counter(r["method"] for r in records)),
            "providers": dict(Counter(r.get("provider") or "unknown" for r in records)),
            "personas": dict(Counter(r.get("persona", "default") for r in records)),
            "latency_ms": {
                "avg": sum(totals) / len(totals) if totals else 0,
                "p95": totals[int(0.95 * (len(totals) - 1))] if totals else 0,
            },
            "recent": [
                {"time": r["timestamp"], "question": r["question"], "in_pdf": r["in_pdf"],
                 "method": r["method"], "provider": r.get("provider"), "total_ms": r["timings_ms"].get("total_ms", 0)}
                for r in records[-15:][::-1]
            ],
            "llm": p.llm.stats,
            "gate": {"version": p.gate.version, **p.gate.metrics} if p.gate else None,
        }

    @app.get("/metrics")
    def prometheus_metrics():
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/graph")
    def graph():
        return Response(get_pipeline().graph_mermaid(), media_type="text/plain")

    return app


app = create_app()
