from config import settings
from src.graph.build import build_rag_graph
from src.graph.nodes import RAGNodes
from src.models import Chunk, RetrievedChunk
from src.pipeline.scope_checker import ScopeChecker


class FakeRetriever:
    """Returns one chunk whose rerank score depends on the query text."""

    def __init__(self, scores_by_query: dict[str, float]):
        self.scores_by_query = scores_by_query
        self.queries: list[str] = []

    def retrieve(self, query, timings=None):
        self.queries.append(query)
        score = self.scores_by_query.get(query, 0.0)
        return [RetrievedChunk(chunk=Chunk("id", "text", "doc.pdf", 1), rerank_score=score)]


class StubLLM:
    """Answers the grounding check with `grounded`, the rewrite with `rewrite`, anything else with 'answer'."""

    def __init__(self, rewrite="better query", grounded="NO"):
        self.rewrite, self.grounded = rewrite, grounded
        self.grounding_questions: list[str] = []

    def complete(self, system_prompt, user_prompt, temperature=None):
        if "search queries" in system_prompt:
            return self.rewrite
        if "YES if the excerpts" in system_prompt:
            self.grounding_questions.append(user_prompt)
            return self.grounded
        return "answer"


def run(retriever, llm, question="original question"):
    graph = build_rag_graph(RAGNodes(retriever, ScopeChecker(llm, high=0.6, low=0.05), llm))
    state = {"question": question, "search_query": question, "rewritten_queries": [], "timings_ms": {}}
    path = []
    for update in graph.stream(state, stream_mode="updates"):
        for node, changes in update.items():
            path.append(node)
            state.update(changes)
    return path, state


def test_strong_match_answers_from_pdf_directly():
    path, state = run(FakeRetriever({"original question": 0.9}), StubLLM())
    assert path == ["retrieve", "check_scope", "generate_from_pdf"]
    assert state["verdict"].in_pdf and state["rewritten_queries"] == []


def test_unrelated_question_skips_retry():
    # Score below retry_min_score: rewording can't help, so no wasted LLM call.
    path, state = run(FakeRetriever({"original question": 0.0}), StubLLM())
    assert path == ["retrieve", "check_scope", "generate_general"]
    assert not state["verdict"].in_pdf


def test_weak_match_retries_and_recovers():
    retriever = FakeRetriever({"original question": 0.02, "better query": 0.9})
    path, state = run(retriever, StubLLM(rewrite="better query"))
    assert path == ["retrieve", "check_scope", "rewrite_query", "retrieve", "check_scope", "generate_from_pdf"]
    assert retriever.queries == ["original question", "better query"]
    assert state["rewritten_queries"] == ["better query"]
    assert state["verdict"].in_pdf


def test_retry_is_capped_then_answers_generally():
    retriever = FakeRetriever({"original question": 0.02, "better query": 0.02})
    path, _ = run(retriever, StubLLM(rewrite="better query"))
    assert path.count("rewrite_query") == settings.max_rewrites
    assert path[-1] == "generate_general"


def test_scope_is_judged_against_original_question_after_rewrite():
    # Borderline score after the rewrite -> the grounding check must see the ORIGINAL question.
    llm = StubLLM(rewrite="better query", grounded="YES")
    retriever = FakeRetriever({"original question": 0.02, "better query": 0.3})
    path, state = run(retriever, llm)
    assert state["verdict"].in_pdf
    assert "Question: original question" in llm.grounding_questions[-1]
    assert "better query" not in llm.grounding_questions[-1]


def test_timings_accumulate_across_retry():
    retriever = FakeRetriever({"original question": 0.02, "better query": 0.9})
    _, state = run(retriever, StubLLM(rewrite="better query"))
    assert {"scope_ms", "rewrite_ms", "llm_ms"} <= state["timings_ms"].keys()
