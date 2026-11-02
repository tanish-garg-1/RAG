from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.models import Chunk, RAGResult, RetrievedChunk, ScopeVerdict, WebResult


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
    def __init__(self, chunks=3, fail=False, web=False):
        self.web = web
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
        if self.web:
            return RAGResult(question, "Canberra [1].", ScopeVerdict(False, 0.0, "low_score", "weak"), [chunk],
                             {"total_ms": 12.0}, [], ["retrieve", "check_scope", "web_search", "generate_general"], "groq", persona,
                             web_sources=[WebResult("Canberra", "https://a.example/c", "Canberra is the capital.")])
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


def test_chat_rejects_whitespace_only_question():
    c, _ = client(FakePipeline())
    assert c.post("/chat", json={"question": "   \n "}).status_code == 422


class IngestingPipeline(FakePipeline):
    """Uses the real PDF loader so corrupt files fail the same way they do in production."""

    def ingest(self, path, describe_images=False):
        from src.ingestion.pdf_loader import PDFLoader
        pages = PDFLoader().load(path)
        return {"source": Path(path).name, "pages": len(pages), "chunks": 1, "image_chunks": 0,
                "seconds": 0.0, "total_chunks": 1}


def test_ingest_unreadable_pdf_is_422_and_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.api.main.settings.pdf_dir", tmp_path)
    c, _ = client(IngestingPipeline())
    for payload in (b"this is not a pdf", b""):
        response = c.post("/ingest", files={"file": ("bad.pdf", payload, "application/pdf")})
        assert response.status_code == 422 and "not a readable PDF" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_ingest_success_saves_file_and_sanitises_the_name(tmp_path, monkeypatch):
    monkeypatch.setattr("src.api.main.settings.pdf_dir", tmp_path)
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "hello world")
    c, _ = client(IngestingPipeline())
    response = c.post("/ingest", files={"file": ("../../evil.pdf", doc.tobytes(), "application/pdf")})
    assert response.status_code == 200 and response.json()["source"] == "evil.pdf"
    assert [p.name for p in tmp_path.iterdir()] == ["evil.pdf"]


def test_chat_reports_where_the_answer_came_from():
    c, _ = client(FakePipeline())
    assert c.post("/chat", json={"question": "q"}).json()["answer_source"] == "pdf"

    c, _ = client(FakePipeline(web=True))
    body = c.post("/chat", json={"question": "capital of Australia?"}).json()
    assert body["answer_source"] == "web" and body["in_pdf"] is False
    assert body["web_sources"][0]["url"] == "https://a.example/c" and body["web_search_error"] is None
    assert 'rag_web_searches_total{outcome="ok"} 1.0' in c.get("/metrics").text
