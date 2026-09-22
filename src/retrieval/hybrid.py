from config import settings
from src.models import Chunk, RetrievedChunk


def reciprocal_rank_fusion(
    dense_results: list[tuple[Chunk, float]],
    bm25_results: list[tuple[Chunk, float]],
    k: int = settings.rrf_k,
) -> list[RetrievedChunk]:
    """Merge two ranked lists. Each list contributes 1 / (k + rank) per chunk.

    Only ranks matter, so the incompatible score scales (cosine 0..1 vs
    unbounded BM25) never have to be normalised against each other. The raw
    scores are kept on each result for analytics.
    """
    merged: dict[str, RetrievedChunk] = {}

    for rank, (chunk, score) in enumerate(dense_results, start=1):
        rc = merged.setdefault(chunk.chunk_id, RetrievedChunk(chunk=chunk))
        rc.dense_score, rc.dense_rank = score, rank
        rc.rrf_score += 1.0 / (k + rank)

    for rank, (chunk, score) in enumerate(bm25_results, start=1):
        rc = merged.setdefault(chunk.chunk_id, RetrievedChunk(chunk=chunk))
        rc.bm25_score, rc.bm25_rank = score, rank
        rc.rrf_score += 1.0 / (k + rank)

    return sorted(merged.values(), key=lambda rc: rc.rrf_score, reverse=True)
