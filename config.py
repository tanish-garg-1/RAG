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

        # --- Backup LLM (Azure OpenAI). Used only when Groq fails; all four values must be set. ---
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        self.llm_timeout_s = 60

        # --- Web search (Tavily). Used only for questions the PDF does not cover; blank key = disabled. ---
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        self.tavily_base_url = os.getenv("TAVILY_BASE_URL", "https://api.tavily.com")   # overridable so tests can use a local fake
        self.web_max_results = 4
        self.web_timeout_s = 15
        self.web_fresh_days = 30   # how far back news is searched for time-sensitive questions

        # --- Local models (FastEmbed / ONNX) ---
        self.embed_model = "BAAI/bge-small-en-v1.5"
        self.rerank_model = "Xenova/ms-marco-MiniLM-L-12-v2"

        # --- Chunking (measured in words) ---
        self.chunk_size = 200
        self.chunk_overlap = 40
        self.min_page_words = 2      # skip pages like chapter dividers that extract to a single digit

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

        # --- Answerability gate (trained classifier that replaces the fixed score thresholds) ---
        # p(in PDF) >= gate_allow -> in PDF, <= gate_deny -> out of PDF, in between -> ask the LLM (as before).
        self.gate_version = os.getenv("GATE_VERSION", "v1")
        self.gate_allow = 0.75
        self.gate_deny = 0.25

        # --- Vision (image / scanned-page understanding at ingest time) ---
        # Gemini is tried first; when its quota runs out, a local Ollama model takes over.
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen3.8-flash-next:125b-mlx")
        self.ollama_timeout = 300          # local models are slow; seconds
        self.min_image_px = 150            # skip icons/bullets smaller than this in either dimension
        self.page_render_dpi = 150         # resolution when a whole scanned page is sent as an image

        # --- Paths ---
        self.pdf_dir = BASE_DIR / "data" / "pdfs"
        self.vector_store_dir = BASE_DIR / "data" / "vector_store"
        self.model_cache_dir = BASE_DIR / "data" / "models"
        self.bm25_path = self.vector_store_dir / "bm25_index.pkl"
        self.query_log_path = BASE_DIR / "logs" / "queries.jsonl"
        self.collection_name = "pdf_chunks"
        self.vision_cache_path = BASE_DIR / "data" / "vision_cache.json"
        self.gate_dir = BASE_DIR / "models" / "gate" / self.gate_version
        self.gate_model_path = self.gate_dir / "gate.joblib"
        self.gate_data_dir = BASE_DIR / "data" / "gate"

    @property
    def web_search_enabled(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def azure_enabled(self) -> bool:
        return bool(self.azure_api_key and self.azure_endpoint and self.azure_deployment)


settings = Settings()
