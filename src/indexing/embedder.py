from config import settings


class Embedder:
    """FastEmbed wrapper. The ONNX model loads on first use, so commands that
    don't need embeddings (like `stats`) start instantly."""

    def __init__(self, model_name: str = settings.embed_model):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(self.model_name, cache_dir=str(settings.model_cache_dir))
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # bge models expect a query instruction prefix; query_embed adds it.
        return next(iter(self.model.query_embed(text))).tolist()
