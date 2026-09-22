from src.models import Chunk, RetrievedChunk
from src.pipeline.scope_checker import ScopeChecker


class StubLLM:
    def __init__(self, reply="YES", error=None):
        self.reply, self.error, self.calls = reply, error, 0

    def complete(self, system_prompt, user_prompt, temperature=None):
        self.calls += 1
        if self.error:
            raise self.error
        return self.reply


def retrieved(score):
    return [RetrievedChunk(chunk=Chunk("id", "text", "doc.pdf", 1), rerank_score=score, rrf_score=0.5)]


def test_high_score_is_in_pdf_without_llm_call():
    llm = StubLLM()
    verdict = ScopeChecker(llm, high=0.6, low=0.05).check("q", retrieved(0.9))
    assert verdict.in_pdf and verdict.method == "high_score" and llm.calls == 0


def test_low_score_is_out_of_pdf_without_llm_call():
    llm = StubLLM()
    verdict = ScopeChecker(llm, high=0.6, low=0.05).check("q", retrieved(0.01))
    assert not verdict.in_pdf and verdict.method == "low_score" and llm.calls == 0


def test_borderline_asks_llm_yes():
    llm = StubLLM("YES")
    verdict = ScopeChecker(llm, high=0.6, low=0.05).check("q", retrieved(0.3))
    assert verdict.in_pdf and verdict.method == "llm_check" and llm.calls == 1


def test_borderline_asks_llm_no():
    verdict = ScopeChecker(StubLLM("No."), high=0.6, low=0.05).check("q", retrieved(0.3))
    assert not verdict.in_pdf and verdict.method == "llm_check"


def test_llm_failure_falls_back_to_midpoint():
    checker = ScopeChecker(StubLLM(error=RuntimeError("down")), high=0.6, low=0.1)
    assert checker.check("q", retrieved(0.4)).in_pdf            # above midpoint 0.35
    assert not checker.check("q", retrieved(0.2)).in_pdf        # below midpoint
    assert checker.check("q", retrieved(0.2)).method == "llm_check_failed"


def test_no_chunks_is_out_of_pdf():
    verdict = ScopeChecker(StubLLM()).check("q", [])
    assert not verdict.in_pdf and verdict.method == "no_results"


def test_decision_ignores_rrf_score():
    # High RRF but low relevance must still be judged out of the PDF.
    chunks = [RetrievedChunk(chunk=Chunk("id", "t", "doc.pdf", 1), rrf_score=0.99, rerank_score=0.01)]
    assert not ScopeChecker(StubLLM(), high=0.6, low=0.05).check("q", chunks).in_pdf
