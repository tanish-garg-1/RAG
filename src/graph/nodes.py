import time

from config import settings
from src.generation.prompts import (
    IN_PDF_PROMPT, OUT_OF_PDF_PROMPT, REWRITE_QUERY_PROMPT, WEB_PROMPT,
    in_pdf_user_prompt, rewrite_user_prompt, web_user_prompt, with_persona,
)
from src.graph.state import RAGState, add_timing
from src.search.web_search import needs_fresh_results, refers_to_document


class RAGNodes:
    """Each method is one graph node: it reads the state and returns only the keys it changes."""

    def __init__(self, retriever, scope_checker, llm, web_search=None):
        self.retriever = retriever
        self.scope_checker = scope_checker
        self.llm = llm
        self.web_search_client = web_search   # None = web search disabled, the graph then has no web_search node

    def retrieve(self, state: RAGState) -> dict:
        timings = dict(state.get("timings_ms", {}))
        step: dict[str, float] = {}
        chunks = self.retriever.retrieve(state.get("search_query") or state["question"], step)
        for key, ms in step.items():
            timings[key] = timings.get(key, 0.0) + ms
        return {"chunks": chunks, "timings_ms": timings}

    def check_scope(self, state: RAGState) -> dict:
        start = time.perf_counter()
        # Always judged against the ORIGINAL question: a rewrite may improve the
        # search, but must never change what the user actually asked.
        verdict = self.scope_checker.check(state["question"], state["chunks"])
        return {"verdict": verdict, "timings_ms": add_timing(state, "scope_ms", (time.perf_counter() - start) * 1000)}

    def rewrite_query(self, state: RAGState) -> dict:
        start = time.perf_counter()
        previous = state.get("rewritten_queries", [])
        query = self.llm.complete(REWRITE_QUERY_PROMPT, rewrite_user_prompt(state["question"], previous), temperature=0)
        query = query.strip().strip('"').splitlines()[0] if query.strip() else state["question"]
        return {
            "search_query": query,
            "rewritten_queries": previous + [query],
            "timings_ms": add_timing(state, "rewrite_ms", (time.perf_counter() - start) * 1000),
        }

    def generate_from_pdf(self, state: RAGState) -> dict:
        start = time.perf_counter()
        answer = self.llm.complete(
            with_persona(IN_PDF_PROMPT, state.get("persona")), in_pdf_user_prompt(state["question"], state["chunks"])
        )
        return {"answer": answer, "timings_ms": add_timing(state, "llm_ms", (time.perf_counter() - start) * 1000)}

    def web_search(self, state: RAGState) -> dict:
        start = time.perf_counter()
        if refers_to_document(state["question"]):
            return {"web_sources": [], "web_error": "Web search skipped: the question is about the PDF itself, which the web cannot know."}
        try:
            question = state["question"]   # the original question, not a PDF-oriented rewrite
            results = self.web_search_client.search(question, fresh=needs_fresh_results(question))
        except Exception as exc:   # deliberately broad: any search failure must degrade to the model's own knowledge
            return {"web_sources": [], "web_error": f"{type(exc).__name__}: {str(exc)[:150]}",
                    "timings_ms": add_timing(state, "web_ms", (time.perf_counter() - start) * 1000)}
        update = {"web_sources": results, "timings_ms": add_timing(state, "web_ms", (time.perf_counter() - start) * 1000)}
        if not results:
            update["web_error"] = "Web search returned no usable results."
        return update

    def generate_general(self, state: RAGState) -> dict:
        start = time.perf_counter()
        if state.get("web_sources"):
            system, user = WEB_PROMPT, web_user_prompt(state["question"], state["web_sources"])
        else:
            system, user = OUT_OF_PDF_PROMPT, state["question"]
        answer = self.llm.complete(with_persona(system, state.get("persona")), user)
        return {"answer": answer, "timings_ms": add_timing(state, "llm_ms", (time.perf_counter() - start) * 1000)}


def route_after_scope(state: RAGState, web_enabled: bool = False) -> str:
    """The conditional edge: answer from the PDF, retry the search, or answer generally (via the web when enabled)."""
    verdict = state["verdict"]
    if web_enabled and needs_fresh_results(state["question"]):
        return "generate_general"   # real-time / news questions: a PDF can't know, even if the topic overlaps with it
    if verdict.in_pdf:
        return "generate_from_pdf"
    retries_left = len(state.get("rewritten_queries", [])) < settings.max_rewrites
    # A near-zero score means the topic simply isn't in the PDF; rewording won't help.
    if retries_left and verdict.top_score >= settings.retry_min_score:
        return "rewrite_query"
    return "generate_general"
