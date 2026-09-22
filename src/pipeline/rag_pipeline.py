import time
from pathlib import Path

from src.generation.groq_client import GroqClient
from src.generation.prompts import OUT_OF_PDF_PROMPT, IN_PDF_PROMPT, in_pdf_user_prompt
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

    def answer(self, question: str) -> RAGResult:
        timings: dict[str, float] = {}
        scope_checker = self.scope_checker  # create the Groq client up front so setup isn't timed as "scope"
        total_start = time.perf_counter()

        chunks = self.retriever.retrieve(question, timings)

        start = time.perf_counter()
        verdict = scope_checker.check(question, chunks)
        timings["scope_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        if verdict.in_pdf:
            answer = self.groq_client.complete(IN_PDF_PROMPT, in_pdf_user_prompt(question, chunks))
        else:
            answer = self.groq_client.complete(OUT_OF_PDF_PROMPT, question)
        timings["llm_ms"] = (time.perf_counter() - start) * 1000
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000

        return RAGResult(question=question, answer=answer, verdict=verdict, chunks=chunks, timings_ms=timings)

    def reset(self):
        self.vector_store.reset()
        self.bm25_store.reset()
