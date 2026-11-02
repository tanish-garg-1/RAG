from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int
    kind: str = "text"   # "text" for PDF text, "image" for a vision-model description of a figure/scanned page


@dataclass
class PageImage:
    """An image pulled from a PDF, ready to be sent to a vision model."""
    page: int
    index: int           # position within the page, for stable chunk ids
    data: bytes
    mime_type: str
    is_full_page: bool   # True when a scanned page was rendered whole (needs transcription, not description)


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
class WebResult:
    """One web search hit that an out-of-PDF answer was written from."""
    title: str
    url: str
    content: str


@dataclass
class ScopeVerdict:
    in_pdf: bool
    top_score: float
    method: str   # "high_score" | "low_score" | "gate_high" | "gate_low" | "llm_check" | "llm_check_failed" | "no_results"
    reason: str
    gate_proba: float | None = None   # P(in PDF) from the trained gate, when one is loaded


@dataclass
class RAGResult:
    question: str
    answer: str
    verdict: ScopeVerdict
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)
    rewritten_queries: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)   # graph nodes in the order they ran
    provider: str | None = None                      # which LLM provider wrote the answer ("groq" / "azure")
    persona: str = "default"
    web_sources: list[WebResult] = field(default_factory=list)   # filled when the answer came from web search
    web_error: str | None = None                                  # why web search was tried but not used

    @property
    def answer_source(self) -> str:
        """Where the answer's facts came from: the "pdf", the "web", or the "model"'s own knowledge."""
        if self.verdict.in_pdf:
            return "pdf"
        return "web" if self.web_sources else "model"

    @property
    def pages(self) -> list[int]:
        return sorted({rc.chunk.page for rc in self.chunks})
