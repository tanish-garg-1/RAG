from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int


@dataclass
class RetrievedChunk:
    chunk: Chunk
    dense_score: float | None = None   # cosine similarity, None if dense search missed it
    dense_rank: int | None = None
    bm25_score: float | None = None    # raw BM25, None if keyword search missed it
    bm25_rank: int | None = None
    rrf_score: float = 0.0             # ranking artifact only, never used for scope decisions
    rerank_score: float | None = None  # cross-encoder relevance, 0..1


@dataclass
class ScopeVerdict:
    in_pdf: bool
    top_score: float
    method: str   # "high_score" | "low_score" | "llm_check" | "llm_check_failed" | "no_results"
    reason: str


@dataclass
class RAGResult:
    question: str
    answer: str
    verdict: ScopeVerdict
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def pages(self) -> list[int]:
        return sorted({rc.chunk.page for rc in self.chunks})
