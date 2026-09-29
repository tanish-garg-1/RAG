import base64
import re

import httpx

from config import settings
from src.models import PageImage

DESCRIBE_PROMPT = (
    "This image comes from a PDF document. If it is a chart, graph, diagram, table, map, screenshot or "
    "photo that carries information, describe it so that someone could answer questions about it without "
    "seeing it: say what kind of image it is, give any title, axis labels, legend and category names, "
    "copy all visible numbers and text exactly, and state the main takeaway.\n"
    "If it is purely decorative (a logo, divider, background, icon or blank), reply with exactly: NONE"
)

TRANSCRIBE_PROMPT = (
    "This is a scanned page from a document. Transcribe all of its text faithfully, in reading order. "
    "Describe any figure or image briefly in [square brackets] where it appears.\n"
    "If the page is blank or has no meaningful content, reply with exactly: NONE"
)


class QuotaExceeded(Exception):
    """The provider's rate or usage limit was hit, so the next provider should take over."""


class GeminiVision:
    name = "gemini"

    def __init__(self, api_key: str = settings.gemini_api_key, model: str = settings.gemini_model):
        from google import genai  # imported here: it is a heavy import and only needed when images are described
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def describe(self, image: PageImage, prompt: str) -> str:
        from google.genai import errors, types
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[types.Part.from_bytes(data=image.data, mime_type=image.mime_type), prompt],
            )
        except errors.APIError as exc:
            if exc.code == 429:
                raise QuotaExceeded(str(exc)) from exc
            raise
        return response.text or ""


class OllamaVision:
    name = "ollama"

    def __init__(self, host: str = settings.ollama_host, model: str = settings.ollama_model,
                 timeout: float = settings.ollama_timeout):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def installed_models(self) -> list[str] | None:
        """Model names Ollama has pulled, or None when the server is not reachable."""
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=3)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except (httpx.HTTPError, ValueError):
            return None

    def describe(self, image: PageImage, prompt: str) -> str:
        response = httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image.data).decode("ascii")],
                }],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        text = response.json()["message"]["content"]
        # Reasoning models can prepend a <think> block; it is not part of the description.
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
