from typing import TypedDict

from src.models import RetrievedChunk, ScopeVerdict


class RAGState(TypedDict, total=False):
    question: str                  # the user's original question, never modified
    search_query: str              # what retrieval actually searches for (rewritten on retry)
    rewritten_queries: list[str]   # every rewrite, in order
    chunks: list[RetrievedChunk]
    verdict: ScopeVerdict
    answer: str
    timings_ms: dict[str, float]


def add_timing(state: RAGState, key: str, ms: float) -> dict[str, float]:
    """Timings accumulate, because a retry runs retrieve and check_scope twice."""
    timings = dict(state.get("timings_ms", {}))
    timings[key] = timings.get(key, 0.0) + ms
    return timings
