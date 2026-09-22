import re

from config import settings
from src.models import Chunk


class Chunker:
    """Splits page text into overlapping word windows. Chunks never cross a page,
    so every chunk keeps an exact page number for citations."""

    def __init__(self, chunk_size: int = settings.chunk_size, overlap: int = settings.chunk_overlap):
        if overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, pages: list[tuple[int, str]], source: str) -> list[Chunk]:
        chunks = []
        step = self.chunk_size - self.overlap

        for page_number, text in pages:
            words = re.sub(r"\s+", " ", text).strip().split(" ")
            if words == [""]:
                continue

            start, index = 0, 0
            while True:
                window = words[start:start + self.chunk_size]
                chunks.append(Chunk(
                    chunk_id=f"{source}::p{page_number}::c{index}",
                    text=" ".join(window),
                    source=source,
                    page=page_number,
                ))
                if start + self.chunk_size >= len(words):
                    break
                start += step
                index += 1

        return chunks
