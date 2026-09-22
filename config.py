import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Quiet Hugging Face download chatter from FastEmbed (Windows can't symlink the cache without admin).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

BASE_DIR = Path(__file__).resolve().parent


class Settings:
    def __init__(self):
        # --- LLM (Groq) ---
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.llm_temperature = 0.2

        # --- Local models (FastEmbed / ONNX) ---
        self.embed_model = "BAAI/bge-small-en-v1.5"
        self.rerank_model = "Xenova/ms-marco-MiniLM-L-12-v2"

        # --- Chunking (measured in words) ---
        self.chunk_size = 200
        self.chunk_overlap = 40

        # --- Retrieval ---
        self.dense_k = 20            # candidates from vector search
        self.bm25_k = 20             # candidates from keyword search
        self.rrf_k = 60              # RRF smoothing constant
        self.rerank_candidates = 10  # fused candidates sent to the reranker
        self.final_k = 4             # chunks given to the LLM

        # --- Scope check (applies to the reranker score, 0..1) ---
        # Above HIGH -> in PDF. Below LOW -> out of PDF. Between -> ask the LLM.
        self.high_confidence = 0.6
        self.low_confidence = 0.05

        # --- Corrective retry (LangGraph loop) ---
        # When a question looks out of the PDF but some chunk scored at least this much,
        # rewrite the question and search again. Near-zero scores skip the retry.
        self.max_rewrites = 1
        self.retry_min_score = 0.005

        # --- Paths ---
        self.pdf_dir = BASE_DIR / "data" / "pdfs"
        self.vector_store_dir = BASE_DIR / "data" / "vector_store"
        self.model_cache_dir = BASE_DIR / "data" / "models"
        self.bm25_path = self.vector_store_dir / "bm25_index.pkl"
        self.query_log_path = BASE_DIR / "logs" / "queries.jsonl"
        self.collection_name = "pdf_chunks"


settings = Settings()
