import logging
from typing import Callable

from src.generation.azure_client import AzureClient
from src.generation.groq_client import GroqClient

log = logging.getLogger("rag.llm")


class LLMRouter:
    """Tries each provider in order and returns the first successful completion.

    Groq is primary. If it raises for ANY reason (rate limit, timeout, outage, bad key),
    the same request is sent to the next provider (Azure OpenAI). Providers are built lazily,
    so an unconfigured backup costs nothing and never blocks startup.

    It exposes the same `complete()` / `last_usage` interface as a single client, so the rest of
    the code (graph nodes, scope checker) does not know there are several providers.
    """

    def __init__(self, providers: list[tuple[str, Callable[[], object]]] | None = None):
        self.providers = providers or [("groq", GroqClient), ("azure", AzureClient)]
        self._clients: dict[str, object] = {}
        self.last_provider: str | None = None
        self.stats = {
            "calls": 0,
            "fallbacks": 0,       # calls answered by a provider other than the first
            "failures": 0,        # calls where every provider failed
            "by_provider": {name: {"ok": 0, "error": 0} for name, _ in self.providers},
        }

    def _client(self, name: str, factory: Callable[[], object]):
        if name not in self._clients:
            self._clients[name] = factory()   # raises RuntimeError when the provider has no credentials
        return self._clients[name]

    def complete(self, system_prompt: str, user_prompt: str, temperature: float | None = None) -> str:
        self.stats["calls"] += 1
        kwargs = {} if temperature is None else {"temperature": temperature}
        errors: list[str] = []

        for position, (name, factory) in enumerate(self.providers):
            try:
                client = self._client(name, factory)
                text = client.complete(system_prompt, user_prompt, **kwargs)
            except Exception as exc:   # deliberately broad: every provider failure should trigger the fallback
                self.stats["by_provider"][name]["error"] += 1
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                log.warning("LLM provider %s failed (%s); trying next", name, type(exc).__name__)
                continue

            self.stats["by_provider"][name]["ok"] += 1
            if position > 0:
                self.stats["fallbacks"] += 1
            self.last_provider = name
            return text

        self.stats["failures"] += 1
        raise RuntimeError("All LLM providers failed. " + " | ".join(errors))

    @property
    def last_usage(self) -> dict[str, int]:
        client = self._clients.get(self.last_provider or "")
        return getattr(client, "last_usage", {}) if client else {}
