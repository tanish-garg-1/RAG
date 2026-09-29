import hashlib
import json
import time
from collections import Counter

from config import settings
from src.models import PageImage
from src.vision.providers import (
    DESCRIBE_PROMPT, TRANSCRIBE_PROMPT, GeminiVision, OllamaVision, QuotaExceeded,
)


class VisionDescriber:
    """Turns an image into text using an ordered list of providers.

    A provider that hits its quota is skipped for `cooldown_s` seconds and the next
    provider takes over; after the cooldown the first provider is tried again. Any other
    error only skips that provider for the one image. Results are cached by image hash,
    so re-running ingest never pays for the same image twice.
    """

    def __init__(self, providers: list, cache_path=settings.vision_cache_path,
                 cooldown_s: float = 60.0, clock=time.monotonic):
        self.providers = providers
        self.cache_path = cache_path
        self.cooldown_s = cooldown_s
        self.clock = clock
        self.cache: dict[str, str] = self._load_cache()
        self.provider_counts: Counter = Counter()
        self.cache_hits = 0
        self.events: list[str] = []          # human-readable notes, e.g. "gemini quota hit; using ollama"
        self.last_error: str | None = None
        self._blocked_until: dict[str, float] = {}

    def describe(self, image: PageImage) -> str | None:
        """Returns the text, "" for a decorative image, or None if every provider failed."""
        key = self._cache_key(image)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]

        if not self.providers:
            self.last_error = "no vision provider available"
            return None

        prompt = TRANSCRIBE_PROMPT if image.is_full_page else DESCRIBE_PROMPT
        for provider in self.providers:
            if self._blocked_until.get(provider.name, 0) > self.clock():
                continue
            try:
                raw = provider.describe(image, prompt)
            except QuotaExceeded:
                self._blocked_until[provider.name] = self.clock() + self.cooldown_s
                self.events.append(f"{provider.name} quota exhausted; falling back to the next provider")
                continue
            except Exception as exc:
                self.last_error = f"{provider.name}: {exc}"
                continue

            text = raw.strip()
            if not text:
                self.last_error = f"{provider.name}: empty response"
                continue
            self.provider_counts[provider.name] += 1
            result = "" if text.upper().rstrip(".") == "NONE" else text
            self.cache[key] = result
            self._save_cache()
            return result
        return None

    @staticmethod
    def _cache_key(image: PageImage) -> str:
        return hashlib.sha256(image.data).hexdigest() + ("|scan" if image.is_full_page else "|fig")

    def _load_cache(self) -> dict[str, str]:
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")


def vision_status() -> dict:
    """What is configured and reachable, for `vision-status` and for ingest to decide what to use."""
    status = {
        "gemini": {"configured": bool(settings.gemini_api_key), "model": settings.gemini_model},
        "ollama": {"reachable": False, "model": settings.ollama_model, "model_installed": False, "host": settings.ollama_host},
    }
    installed = OllamaVision().installed_models()
    if installed is not None:
        status["ollama"]["reachable"] = True
        status["ollama"]["model_installed"] = settings.ollama_model in installed
        status["ollama"]["installed"] = installed
    return status


def build_describer() -> VisionDescriber | None:
    """Gemini first (if a key is set), then Ollama (if its server is running).

    With no provider the describer is still returned when the cache has entries, so re-ingesting
    keeps every figure that was already described. None only when there is nothing to work with.
    """
    providers = []
    if settings.gemini_api_key:
        providers.append(GeminiVision())
    if OllamaVision().installed_models() is not None:
        providers.append(OllamaVision())
    describer = VisionDescriber(providers)
    return describer if providers or describer.cache else None
