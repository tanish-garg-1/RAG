import time
from pathlib import Path
from typing import Callable

from src.generation.groq_client import GroqClient
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


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()
        if not self.bm25_store.load() and self.vector_store.count() > 0:
            self.rebuild_bm25()
        self.retriever = Retriever(self.embedder, self.vector_store, self.bm25_store)
        self._groq_client = None
        self._scope_checker = None
        self._graph = None

    # The Groq client is created lazily so `ingest` works without an API key.
    @property
    def groq_client(self) -> GroqClient:
        if self._groq_client is None:
            self._groq_client = GroqClient()
        return self._groq_client

    @property
    def scope_checker(self) -> ScopeChecker:
        if self._scope_checker is None:
            self._scope_checker = ScopeChecker(self.groq_client)
        return self._scope_checker

    def ingest(self, pdf_path: str) -> dict:
        source = Path(pdf_path).name
        start = time.perf_counter()

        pages = PDFLoader().load(pdf_path)
        chunks = Chunker().split(pages, source)
        if not chunks:
            raise ValueError(f"No extractable text in {source}. Is it a scanned PDF?")

        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.vector_store.delete_source(source)  # re-ingesting replaces, never duplicates
        self.vector_store.add_chunks(chunks, embeddings)
        self.rebuild_bm25()

        return {
            "source": source,
            "pages": len(pages),
            "empty_pages": sum(1 for _, text in pages if not text.strip()),
            "chunks": len(chunks),
            "total_chunks": self.vector_store.count(),
            "seconds": time.perf_counter() - start,
        }

    def rebuild_bm25(self):
        self.bm25_store.build(self.vector_store.all_chunks())
        self.bm25_store.save()

    @property
    def graph(self):
        if self._graph is None:
            nodes = RAGNodes(self.retriever, self.scope_checker, self.groq_client)
            self._graph = build_rag_graph(nodes)
        return self._graph

    def answer(self, question: str, on_step: Callable[[str], None] | None = None) -> RAGResult:
        """Run the LangGraph workflow. `on_step` is called with each node name as it finishes."""
        graph = self.graph  # build before timing so Groq client setup isn't counted
        total_start = time.perf_counter()

        state: dict = {"question": question, "search_query": question, "rewritten_queries": [], "timings_ms": {}}
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
        )

    def graph_mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def reset(self):
        self.vector_store.reset()
        self.bm25_store.reset()
