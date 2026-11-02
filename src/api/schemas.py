from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Persona = Literal["default", "formal", "friendly", "concise", "eli5"]


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)   # a whitespace-only question becomes empty and is rejected

    question: str = Field(min_length=1, max_length=2000)
    persona: Persona = "default"
    explain: bool = Field(False, description="Include the SHAP breakdown of the in-PDF decision.")


class Source(BaseModel):
    source: str
    page: int
    kind: str
    rerank_score: float | None
    text: str


class WebSource(BaseModel):
    title: str
    url: str
    content: str


class Contribution(BaseModel):
    feature: str
    description: str
    value: float
    shap: float


class Explanation(BaseModel):
    base_value: float
    probability: float
    contributions: list[Contribution]


class ChatResponse(BaseModel):
    answer: str
    in_pdf: bool
    verdict_method: str
    verdict_reason: str
    gate_probability: float | None
    top_score: float
    provider: str | None
    persona: str
    answer_source: Literal["pdf", "web", "model"]   # where the facts came from; the UI banner follows this
    web_sources: list[WebSource] = []
    web_search_error: str | None = None
    path: list[str]
    rewritten_queries: list[str]
    sources: list[Source]
    timings_ms: dict[str, float]
    explanation: Explanation | None = None


class IngestResponse(BaseModel):
    source: str
    pages: int
    chunks: int
    image_chunks: int
    seconds: float
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    gate_loaded: bool
    gate_version: str | None
    groq_configured: bool
    azure_configured: bool
    web_search_configured: bool
