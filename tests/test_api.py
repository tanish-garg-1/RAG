from fastapi.testclient import TestClient

from src.api.main import create_app
from src.models import Chunk, RAGResult, RetrievedChunk, ScopeVerdict


class FakeStore:
    def __init__(self, n):
        self.n = n

    def count(self):
        return self.n

    def all_chunks(self):
        return [Chunk(f"c{i}", "t", "doc.pdf", 1) for i in range(self.n)]


class FakeLLM:
    def __init__(self):
        self.stats = {"calls": 0, "fallbacks": 0, "failures": 0, "by_provider": {}}
        self.last_usage = {}


class FakePipeline:
    def __init__(self, chunks=3, fail=False):
        self.vector_store = FakeStore(chunks)
        self.gate = None
        self.llm = FakeLLM()
        self.fail = fail
        self.personas = []

    def answer(self, question, persona="default"):
        if self.fail:
            raise RuntimeError("All LLM providers failed.")
        self.personas.append(persona)
        chunk = RetrievedChunk(Chunk("c", "Monopolies matter.", "doc.pdf", 7), rerank_score=0.9)
        return RAGResult(question, "Because [p. 7].", ScopeVerdict(True, 0.9, "high_score", "strong"), [chunk],
                         {"total_ms": 12.0}, [], ["retrieve", "check_scope", "generate_from_pdf"], "groq", persona)

    def explain(self, result):
        return None


class NullLogger:
    def __init__(self):
        self.records = []

    def log(self, result, usage=None):
        self.records.append(result.question)

    def read_all(self):
        return []


def client(pipeline):
    logger = NullLogger()
    return TestClient(create_app(pipeline=pipeline, query_logger=logger, warm_up=False)), logger


def test_health_reports_index_and_providers():
    c, _ = client(FakePipeline(chunks=5))
    body = c.get("/health").json()
    assert body["status"] == "ok" and body["chunks_indexed"] == 5 and body["gate_loaded"] is False


def test_chat_returns_answer_sources_and_logs_it():
    pipeline = FakePipeline()
    c, logger = client(pipeline)
    response = c.post("/chat", json={"question": "Why monopolies?", "persona": "eli5"})
    assert response.status_code == 200
    body = response.json()
    assert body["in_pdf"] is True and body["provider"] == "groq"
    assert body["sources"][0]["page"] == 7
    assert pipeline.personas == ["eli5"] and logger.records == ["Why monopolies?"]


def test_chat_rejects_unknown_persona_and_empty_question():
    c, _ = client(FakePipeline())
    assert c.post("/chat", json={"question": "hi", "persona": "pirate"}).status_code == 422
    assert c.post("/chat", json={"question": ""}).status_code == 422


def test_chat_on_empty_index_is_a_conflict():
    c, _ = client(FakePipeline(chunks=0))
    response = c.post("/chat", json={"question": "hi"})
    assert response.status_code == 409 and "Upload a PDF" in response.json()["detail"]


def test_all_providers_failing_is_a_bad_gateway():
    c, _ = client(FakePipeline(fail=True))
    assert c.post("/chat", json={"question": "hi"}).status_code == 502


def test_ingest_rejects_non_pdf():
    c, _ = client(FakePipeline())
    response = c.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_metrics_endpoint_exposes_prometheus_text():
    c, _ = client(FakePipeline())
    c.post("/chat", json={"question": "hi"})
    text = c.get("/metrics").text
    assert 'rag_questions_total{verdict="in_pdf"}' in text and "rag_answer_seconds_bucket" in text
