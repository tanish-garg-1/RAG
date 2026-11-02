import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

from config import settings
from src.gate.gate import AnswerabilityGate
from src.generation.llm_router import LLMRouter
from src.graph.build import build_rag_graph
from src.graph.nodes import RAGNodes
from src.indexing.bm25_store import BM25Store
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore
from src.ingestion.chunker import Chunker
from src.ingestion.pdf_loader import PDFLoader
from src.models import RAGResult
from src.pipeline.scope_checker import ScopeChecker
from src.retrieval.retriever import Retriever
from src.search.web_search import TavilySearch
from src.vision.describer import build_describer


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()
        if not self.bm25_store.load() and self.vector_store.count() > 0:
            self.rebuild_bm25()
        self.retriever = Retriever(self.embedder, self.vector_store, self.bm25_store)
        self._llm = None
        self._gate = None
        self._gate_loaded = False
        self._scope_checker = None
        self._graph = None

    # The router builds its provider clients lazily, so `ingest` works without any API key.
    @property
    def llm(self) -> LLMRouter:
        if self._llm is None:
            self._llm = LLMRouter()
        return self._llm

    groq_client = llm   # old name, still used by main.py's usage logging

    @property
    def gate(self) -> AnswerabilityGate | None:
        """The trained answerability model, or None (then fixed thresholds are used)."""
        if not self._gate_loaded:
            self._gate = AnswerabilityGate.load()
            self._gate_loaded = True
        return self._gate

    @property
    def scope_checker(self) -> ScopeChecker:
        if self._scope_checker is None:
            self._scope_checker = ScopeChecker(self.llm, gate=self.gate)
        return self._scope_checker

    def ingest(self, pdf_path: str, describe_images: bool = True,
               on_progress: Callable[[str], None] | None = None) -> dict:
        source = Path(pdf_path).name
        start = time.perf_counter()

        loader = PDFLoader()
        pages = loader.load(pdf_path)
        chunker = Chunker()
        chunks = chunker.split(pages, source)

        vision = {"enabled": False}
        if describe_images:
            image_chunks, vision = self._describe_images(loader, chunker, pdf_path, source, on_progress)
            chunks += image_chunks

        if not chunks:
            raise ValueError(
                f"No extractable text in {source}. It looks scanned; configure a vision provider "
                "(GEMINI_API_KEY or a running Ollama) and ingest with images enabled."
            )

        if on_progress:
            on_progress("Embedding chunks...")
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.vector_store.delete_source(source)  # re-ingesting replaces, never duplicates
        self.vector_store.add_chunks(chunks, embeddings)
        self.rebuild_bm25()

        return {
            "source": source,
            "pages": len(pages),
            "empty_pages": sum(1 for _, text in pages if not text.strip()),
            "chunks": len(chunks),
            "image_chunks": sum(1 for c in chunks if c.kind == "image"),
            "vision": vision,
            "total_chunks": self.vector_store.count(),
            "seconds": time.perf_counter() - start,
        }

    def _describe_images(self, loader, chunker, pdf_path, source, on_progress):
        """Returns (image chunks, stats). Vision problems are reported in the stats and never abort ingest."""
        describer = build_describer()
        if describer is None:
            return [], {"enabled": False, "reason": "no vision provider available (set GEMINI_API_KEY or start Ollama)"}

        chunks, stats = [], {"enabled": True, "images": 0, "described": 0, "decorative": 0, "failed": 0}
        for image in loader.iter_images(pdf_path):
            stats["images"] += 1
            if on_progress:
                on_progress(f"Describing images ({stats['images']} so far)...")
            text = describer.describe(image)
            if text is None:
                stats["failed"] += 1
            elif not text:
                stats["decorative"] += 1
            else:
                stats["described"] += 1
                label = "Scanned page" if image.is_full_page else "Figure"
                for j, chunk in enumerate(chunker.split([(image.page, f"[{label} on page {image.page}] {text}")], source)):
                    chunks.append(replace(chunk, chunk_id=f"{source}::p{image.page}::img{image.index}::c{j}", kind="image"))

        stats.update(
            providers=dict(describer.provider_counts), cache_hits=describer.cache_hits,
            events=describer.events, last_error=describer.last_error,
        )
        return chunks, stats

    def rebuild_bm25(self):
        self.bm25_store.build(self.vector_store.all_chunks())
        self.bm25_store.save()

    @property
    def graph(self):
        if self._graph is None:
            web = TavilySearch() if settings.web_search_enabled else None
            nodes = RAGNodes(self.retriever, self.scope_checker, self.llm, web)
            self._graph = build_rag_graph(nodes)
        return self._graph

    def answer(self, question: str, on_step: Callable[[str], None] | None = None, persona: str = "default") -> RAGResult:
        """Run the LangGraph workflow. `on_step` is called with each node name as it finishes.
        `persona` picks the answer style (see PERSONAS in src/generation/prompts.py)."""
        graph = self.graph  # build before timing so Groq client setup isn't counted
        total_start = time.perf_counter()

        state: dict = {
            "question": question, "search_query": question, "rewritten_queries": [], "timings_ms": {}, "persona": persona,
        }
        path: list[str] = []
        for update in graph.stream(state, stream_mode="updates"):
            for node, changes in update.items():
                path.append(node)
                state.update(changes)
                if on_step:
                    on_step(node)

        timings = state["timings_ms"]
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000
        return RAGResult(
            question=question,
            answer=state["answer"],
            verdict=state["verdict"],
            chunks=state["chunks"],
            timings_ms=timings,
            rewritten_queries=state["rewritten_queries"],
            path=path,
            provider=self.llm.last_provider,
            persona=persona,
            web_sources=state.get("web_sources", []),
            web_error=state.get("web_error"),
        )

    def explain(self, result: RAGResult) -> dict | None:
        """SHAP breakdown of the in-PDF / out-of-PDF decision, or None when no gate is trained."""
        return self.gate.explain(result.question, result.chunks) if self.gate else None

    def graph_mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def reset(self):
        self.vector_store.reset()
        self.bm25_store.reset()
