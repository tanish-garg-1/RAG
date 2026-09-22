import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from src.models import Chunk


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(settings.vector_store_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # embedding_function=None: we always pass our own FastEmbed vectors.
        self.collection = self.client.get_or_create_collection(
            settings.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 500):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings[i:i + batch_size],
                documents=[c.text for c in batch],
                metadatas=[{"source": c.source, "page": c.page} for c in batch],
            )

    def delete_source(self, source: str):
        self.collection.delete(where={"source": source})

    def dense_search(self, query_embedding: list[float], k: int) -> list[tuple[Chunk, float]]:
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            chunk = Chunk(chunk_id=chunk_id, text=text, source=meta["source"], page=meta["page"])
            hits.append((chunk, 1.0 - distance))  # cosine distance -> similarity
        return hits

    def all_chunks(self) -> list[Chunk]:
        result = self.collection.get(include=["documents", "metadatas"])
        return [
            Chunk(chunk_id=chunk_id, text=text, source=meta["source"], page=meta["page"])
            for chunk_id, text, meta in zip(result["ids"], result["documents"], result["metadatas"])
        ]

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        self.client.delete_collection(settings.collection_name)
