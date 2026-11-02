import re

from src.models import RetrievedChunk

# Order matters: the trained model sees the features in exactly this order.
FEATURE_NAMES = [
    "top_rerank",      # cross-encoder score of the best chunk
    "rerank_gap",      # best score minus second-best (a clear winner vs. a crowd of mediocre chunks)
    "mean_rerank",     # average score of the retrieved chunks
    "n_relevant",      # how many chunks scored above 0.1
    "top_dense",       # embedding cosine similarity of the best chunk
    "max_bm25",        # strongest keyword-match score (0 if keyword search found nothing)
    "top_in_bm25",     # 1 if the best chunk was ALSO found by keyword search
    "q_words",         # question length in words
    "q_overlap",       # share of the question's content words that appear in the best chunk
]

FEATURE_DOCS = {
    "top_rerank": "Relevance of the best passage (cross-encoder)",
    "rerank_gap": "How far the best passage stands above the runner-up",
    "mean_rerank": "Average relevance of the retrieved passages",
    "n_relevant": "Number of passages with relevance above 0.1",
    "top_dense": "Semantic (embedding) similarity of the best passage",
    "max_bm25": "Strongest exact-keyword match",
    "top_in_bm25": "Best passage also found by keyword search",
    "q_words": "Question length (words)",
    "q_overlap": "Question words that appear in the best passage",
}

_STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or but is are was were be been being do does did "
    "what which who whom whose when where why how this that these those it its as about into than then "
    "so such can could should would will shall may might must not no yes i you he she we they me my your "
    "his her our their there here has have had".split()
)


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOPWORDS and len(w) > 1}


def extract_features(question: str, chunks: list[RetrievedChunk]) -> list[float]:
    """Turns one retrieval result into a fixed-length numeric vector (see FEATURE_NAMES)."""
    if not chunks:
        return [0.0] * len(FEATURE_NAMES)

    scores = [rc.rerank_score or 0.0 for rc in chunks]
    top = chunks[0]
    question_words = content_words(question)
    overlap = len(question_words & content_words(top.chunk.text)) / len(question_words) if question_words else 0.0

    return [
        scores[0],
        scores[0] - (scores[1] if len(scores) > 1 else 0.0),
        sum(scores) / len(scores),
        float(sum(s > 0.1 for s in scores)),
        top.dense_score or 0.0,
        max((rc.bm25_score or 0.0) for rc in chunks),
        1.0 if top.bm25_rank else 0.0,
        float(len(question.split())),
        overlap,
    ]
