import re

import httpx

from config import settings
from src.models import WebResult

# Questions about "the book / this PDF / page 5" are about the user's document, which the web cannot know.
# Searching them returns unrelated pages (a search for "tell me about the book" finds random books).
DOCUMENT_REFERENCE = re.compile(
    r"\b(this|the|my|your)\s+(pdf|document|file|manuscript)\b"
    r"|\bthis\s+(book|text|paper|chapter|article|report)\b"
    r"|\bthe\s+(book|text|chapter)\s*[?.!]*$"
    r"|\b(the|this)\s+author\b"
    r"|\b(page|chapter)\s+\d+\b",
    re.IGNORECASE,
)


MIN_FRESH_RESULTS = 3   # fewer news hits than this are topped up with a normal search


def refers_to_document(question: str) -> bool:
    return bool(DOCUMENT_REFERENCE.search(question))


# Questions whose answer changes over time. A PDF can never answer these, so they always go to the web,
# and the search is restricted to recent news instead of evergreen pages.
TIME_SENSITIVE = re.compile(
    r"\b(today|tonight|yesterday|right now|currently|breaking|latest|most recent|recent(ly)?|news|headlines?"
    r"|weather|forecast|score|live|stock price|share price|exchange rate|price of|this (week|month|year)"
    r"|(current|now)\s+(price|rate|ceo|president|prime minister|champion|leader)|202[5-9])\b",
    re.IGNORECASE,
)


def needs_fresh_results(question: str) -> bool:
    """True for real-time / news questions. Questions about the user's own PDF never count."""
    return bool(TIME_SENSITIVE.search(question)) and not refers_to_document(question)


class TavilySearch:
    """Web search through the Tavily API (https://tavily.com), which returns clean text snippets for LLMs.

    Only the user's question is sent, and only when the PDF cannot answer it. No PDF text ever leaves the machine.
    """

    def __init__(self, client: httpx.Client | None = None):
        if not settings.web_search_enabled and client is None:
            raise RuntimeError("Web search is not configured. Set TAVILY_API_KEY in .env to enable it.")
        self.client = client or httpx.Client(timeout=settings.web_timeout_s)

    def search(self, query: str, fresh: bool = False, max_results: int = settings.web_max_results) -> list[WebResult]:
        """Raises on network errors, bad keys or rate limits; the caller decides how to degrade.

        `fresh` searches news from the last `web_fresh_days` days first, which is what real-time questions need.
        If that finds fewer than MIN_FRESH_RESULTS pages, the rest are filled from a normal search.
        """
        if not fresh:
            return self._request(query, max_results)
        results = self._request(query, max_results, topic="news", days=settings.web_fresh_days)
        if len(results) >= MIN_FRESH_RESULTS:
            return results
        # Few news hits (weather, a quiet topic): top up with normal results, newest news first.
        seen = {r.url for r in results}
        extra = [r for r in self._request(query, max_results) if r.url not in seen]
        return (results + extra)[:max_results]

    def _request(self, query: str, max_results: int, **extra) -> list[WebResult]:
        response = self.client.post(
            f"{settings.tavily_base_url}/search",
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            json={"query": query, "max_results": max_results, "search_depth": "basic", "include_answer": False, **extra},
        )
        response.raise_for_status()
        return [
            WebResult(
                title=hit.get("title") or hit["url"], url=hit["url"], content=(hit.get("content") or "").strip(),
                published=hit.get("published_date"),
            )
            for hit in response.json().get("results", [])
            if hit.get("url") and (hit.get("content") or "").strip()
        ]
