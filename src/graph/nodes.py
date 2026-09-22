import time

from config import settings
from src.generation.prompts import (
    IN_PDF_PROMPT, OUT_OF_PDF_PROMPT, REWRITE_QUERY_PROMPT, in_pdf_user_prompt, rewrite_user_prompt,
)
from src.graph.state import RAGState, add_timing


class RAGNodes:
    """Each method is one graph node: it reads the state and returns only the keys it changes."""

    def __init__(self, retriever, scope_checker, llm):
        self.retriever = retriever
        self.scope_checker = scope_checker
        self.llm = llm

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
        answer = self.llm.complete(IN_PDF_PROMPT, in_pdf_user_prompt(state["question"], state["chunks"]))
        return {"answer": answer, "timings_ms": add_timing(state, "llm_ms", (time.perf_counter() - start) * 1000)}

    def generate_general(self, state: RAGState) -> dict:
        start = time.perf_counter()
        answer = self.llm.complete(OUT_OF_PDF_PROMPT, state["question"])
        return {"answer": answer, "timings_ms": add_timing(state, "llm_ms", (time.perf_counter() - start) * 1000)}


def route_after_scope(state: RAGState) -> str:
    """The conditional edge: answer from the PDF, retry the search, or answer generally."""
    verdict = state["verdict"]
    if verdict.in_pdf:
        return "generate_from_pdf"
    retries_left = len(state.get("rewritten_queries", [])) < settings.max_rewrites
    # A near-zero score means the topic simply isn't in the PDF; rewording won't help.
    if retries_left and verdict.top_score >= settings.retry_min_score:
        return "rewrite_query"
    return "generate_general"
