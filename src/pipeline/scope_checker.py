from config import settings
from src.generation.prompts import GROUNDING_CHECK_PROMPT, grounding_user_prompt
from src.models import RetrievedChunk, ScopeVerdict


class ScopeChecker:
    """Decides whether a question is answered by the PDF.

    Reads the top reranker score (never the RRF score, which only encodes rank).
    Clear cases are decided for free; only the borderline band costs an LLM call.

    With a trained AnswerabilityGate the decision is the model's probability P(in PDF);
    without one it falls back to two fixed thresholds on the top reranker score.
    """

    def __init__(self, llm, high: float = settings.high_confidence, low: float = settings.low_confidence, gate=None):
        self.llm = llm
        self.high = high
        self.low = low
        self.gate = gate

    def check(self, question: str, chunks: list[RetrievedChunk]) -> ScopeVerdict:
        if not chunks:
            return ScopeVerdict(False, 0.0, "no_results", "No matching chunks were found in the index.")

        top = chunks[0].rerank_score or 0.0
        proba = self.gate.predict_proba(question, chunks) if self.gate else None

        if proba is not None:
            if proba >= settings.gate_allow:
                return ScopeVerdict(True, top, "gate_high", f"Gate P(in PDF) = {proba:.2f} >= {settings.gate_allow:.2f}.", proba)
            if proba <= settings.gate_deny:
                return ScopeVerdict(False, top, "gate_low", f"Gate P(in PDF) = {proba:.2f} <= {settings.gate_deny:.2f}.", proba)
        else:
            if top >= self.high:
                return ScopeVerdict(True, top, "high_score", f"Top relevance {top:.2f} >= {self.high:.2f}.")
            if top < self.low:
                return ScopeVerdict(False, top, "low_score", f"Top relevance {top:.2f} < {self.low:.2f}.")

        try:
            reply = self.llm.complete(GROUNDING_CHECK_PROMPT, grounding_user_prompt(question, chunks), temperature=0)
        except Exception as exc:
            in_pdf = proba >= 0.5 if proba is not None else top >= (self.high + self.low) / 2
            return ScopeVerdict(in_pdf, top, "llm_check_failed", f"Grounding check failed ({exc}); used the model's best guess.", proba)

        in_pdf = reply.strip().upper().startswith("YES")
        return ScopeVerdict(
            in_pdf, top, "llm_check",
            f"Score was borderline; LLM judged the excerpts {'sufficient' if in_pdf else 'insufficient'}.",
            proba,
        )
