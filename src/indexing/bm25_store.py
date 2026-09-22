import pickle
import re

from rank_bm25 import BM25Okapi

from config import settings
from src.models import Chunk

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "did", "do", "does", "for",
    "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my",
    "not", "of", "on", "or", "our", "so", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


class BM25Store:
    """Keyword index. It is derived data: always rebuilt from the chunks in the
    vector store, so the two indexes cannot drift apart."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def save(self):
        settings.bm25_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings.bm25_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, f)

    def load(self) -> bool:
        if not settings.bm25_path.exists():
            return False
        with open(settings.bm25_path, "rb") as f:
            data = pickle.load(f)
        self.chunks, self.bm25 = data["chunks"], data["bm25"]
        return True

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        tokens = tokenize(query)
        if self.bm25 is None or not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        # A score of 0 means no query term appears in the chunk; that is not a hit.
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def reset(self):
        self.chunks, self.bm25 = [], None
        settings.bm25_path.unlink(missing_ok=True)
