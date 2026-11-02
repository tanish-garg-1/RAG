from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

# A private registry (not the global default) so tests can build several apps without "duplicate metric" errors.
REGISTRY = CollectorRegistry()

QUESTIONS = Counter(
    "rag_questions_total", "Questions answered, by whether the PDF covered them",
    ["verdict"], registry=REGISTRY,
)
LLM_PROVIDER = Counter(
    "rag_llm_answers_total", "Answers by the LLM provider that wrote them",
    ["provider"], registry=REGISTRY,
)
FALLBACKS = Counter("rag_llm_fallbacks_total", "Answers served by the backup provider", registry=REGISTRY)
WEB_SEARCHES = Counter(
    "rag_web_searches_total", "Out-of-PDF questions sent to web search, by outcome",
    ["outcome"], registry=REGISTRY,
)
ERRORS = Counter("rag_errors_total", "Requests that failed", ["endpoint"], registry=REGISTRY)
INGESTS = Counter("rag_ingests_total", "PDFs ingested", registry=REGISTRY)
LATENCY = Histogram(
    "rag_answer_seconds", "End-to-end answer latency",
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 80), registry=REGISTRY,
)


def render() -> bytes:
    return generate_latest(REGISTRY)
