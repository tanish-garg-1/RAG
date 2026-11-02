import httpx

from config import settings
from src.models import WebResult


class TavilySearch:
    """Web search through the Tavily API (https://tavily.com), which returns clean text snippets for LLMs.

    Only the user's question is sent, and only when the PDF cannot answer it. No PDF text ever leaves the machine.
    """

    def __init__(self, client: httpx.Client | None = None):
        if not settings.web_search_enabled and client is None:
            raise RuntimeError("Web search is not configured. Set TAVILY_API_KEY in .env to enable it.")
        self.client = client or httpx.Client(timeout=settings.web_timeout_s)

    def search(self, query: str, max_results: int = settings.web_max_results) -> list[WebResult]:
        """Raises on network errors, bad keys or rate limits; the caller decides how to degrade."""
        response = self.client.post(
            f"{settings.tavily_base_url}/search",
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            json={"query": query, "max_results": max_results, "search_depth": "basic", "include_answer": False},
        )
        response.raise_for_status()
        return [
            WebResult(title=hit.get("title") or hit["url"], url=hit["url"], content=(hit.get("content") or "").strip())
            for hit in response.json().get("results", [])
            if hit.get("url") and (hit.get("content") or "").strip()
        ]
