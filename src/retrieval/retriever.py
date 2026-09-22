import time

from config import settings
from src.indexing.bm25_store import BM25Store
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore
from src.models import RetrievedChunk
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker


class Retriever:
    """dense + BM25 -> RRF -> top-N -> cross-encoder rerank -> top-k"""

    def __init__(self, embedder: Embedder, vector_store: VectorStore, bm25_store: BM25Store):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.reranker = Reranker()

    def retrieve(self, query: str, timings: dict[str, float] | None = None) -> list[RetrievedChunk]:
        timings = timings if timings is not None else {}

        start = time.perf_counter()
        dense = self.vector_store.dense_search(self.embedder.embed_query(query), settings.dense_k)
        timings["dense_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        sparse = self.bm25_store.search(query, settings.bm25_k)
        timings["bm25_ms"] = (time.perf_counter() - start) * 1000

        candidates = reciprocal_rank_fusion(dense, sparse)[:settings.rerank_candidates]

        start = time.perf_counter()
        reranked = self.reranker.rerank(query, candidates)
        timings["rerank_ms"] = (time.perf_counter() - start) * 1000

        return reranked[:settings.final_k]
