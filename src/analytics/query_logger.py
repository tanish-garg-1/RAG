import json
from datetime import datetime, timezone

from config import settings
from src.models import RAGResult


class QueryLogger:
    def __init__(self, path=settings.query_log_path):
        self.path = path

    def log(self, result: RAGResult, usage: dict | None = None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": result.question,
            "in_pdf": result.verdict.in_pdf,
            "method": result.verdict.method,
            "top_rerank_score": round(result.verdict.top_score, 4),
            "pages": result.pages,
            "path": result.path,
            "rewritten_queries": result.rewritten_queries,
            "chunks": [
                {
                    "page": rc.chunk.page,
                    "rerank": rc.rerank_score,
                    "dense": rc.dense_score,
                    "dense_rank": rc.dense_rank,
                    "bm25": rc.bm25_score,
                    "bm25_rank": rc.bm25_rank,
                    "rrf": rc.rrf_score,
                }
                for rc in result.chunks
            ],
            "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
            "usage": usage or {},
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
