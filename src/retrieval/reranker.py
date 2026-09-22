import math

from config import settings
from src.models import RetrievedChunk


class Reranker:
    """Cross-encoder: reads the question and each chunk together and scores
    true relevance. Its output drives the in-PDF / out-of-PDF decision."""

    def __init__(self, model_name: str = settings.rerank_model):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._model = TextCrossEncoder(self.model_name, cache_dir=str(settings.model_cache_dir))
        return self._model

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        logits = self.model.rerank(query, [rc.chunk.text for rc in candidates])
        for rc, logit in zip(candidates, logits):
            # The model emits raw logits; sigmoid maps them to a 0..1 relevance score.
            rc.rerank_score = 1.0 / (1.0 + math.exp(-logit))
        return sorted(candidates, key=lambda rc: rc.rerank_score, reverse=True)
