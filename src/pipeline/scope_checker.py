from config import settings
from src.generation.prompts import GROUNDING_CHECK_PROMPT, grounding_user_prompt
from src.models import RetrievedChunk, ScopeVerdict


class ScopeChecker:
    """Decides whether a question is answered by the PDF.

    Reads the top reranker score (never the RRF score, which only encodes rank).
    Clear cases are decided by threshold for free; only the borderline band
    costs an LLM call.
    """

    def __init__(self, llm, high: float = settings.high_confidence, low: float = settings.low_confidence):
        self.llm = llm
        self.high = high
        self.low = low

    def check(self, question: str, chunks: list[RetrievedChunk]) -> ScopeVerdict:
        if not chunks:
            return ScopeVerdict(False, 0.0, "no_results", "No matching chunks were found in the index.")

        top = chunks[0].rerank_score or 0.0

        if top >= self.high:
            return ScopeVerdict(True, top, "high_score", f"Top relevance {top:.2f} >= {self.high:.2f}.")
        if top < self.low:
            return ScopeVerdict(False, top, "low_score", f"Top relevance {top:.2f} < {self.low:.2f}.")

        try:
            reply = self.llm.complete(GROUNDING_CHECK_PROMPT, grounding_user_prompt(question, chunks), temperature=0)
        except Exception as exc:
            in_pdf = top >= (self.high + self.low) / 2
            return ScopeVerdict(in_pdf, top, "llm_check_failed", f"Grounding check failed ({exc}); used score midpoint.")

        in_pdf = reply.strip().upper().startswith("YES")
        return ScopeVerdict(
            in_pdf, top, "llm_check",
            f"Top relevance {top:.2f} was borderline; LLM judged the excerpts {'sufficient' if in_pdf else 'insufficient'}.",
        )
