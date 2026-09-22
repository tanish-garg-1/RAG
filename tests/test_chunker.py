import pytest

from src.ingestion.chunker import Chunker


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_short_page_is_one_chunk():
    chunks = Chunker(chunk_size=10, overlap=2).split([(1, words(5))], "doc.pdf")
    assert len(chunks) == 1
    assert chunks[0].text == words(5)
    assert chunks[0].page == 1
    assert chunks[0].chunk_id == "doc.pdf::p1::c0"


def test_windows_overlap_and_cover_every_word():
    chunks = Chunker(chunk_size=10, overlap=3).split([(1, words(25))], "doc.pdf")
    texts = [c.text.split() for c in chunks]
    assert [len(t) for t in texts] == [10, 10, 10, 4]
    assert texts[0][-3:] == texts[1][:3]           # overlap carried forward
    assert texts[-1][-1] == "w24"                  # last word included
    assert {w for t in texts for w in t} == set(words(25).split())


def test_exact_fit_does_not_emit_trailing_duplicate():
    chunks = Chunker(chunk_size=10, overlap=2).split([(1, words(10))], "doc.pdf")
    assert len(chunks) == 1


def test_chunks_never_cross_pages_and_skip_empty_pages():
    pages = [(1, words(3, "a")), (2, "   \n  "), (3, words(3, "b"))]
    chunks = Chunker(chunk_size=10, overlap=2).split(pages, "doc.pdf")
    assert [c.page for c in chunks] == [1, 3]
    assert "a0" not in chunks[1].text


def test_whitespace_is_normalised():
    chunks = Chunker(chunk_size=10, overlap=2).split([(1, "hello\n\n  world\tagain")], "doc.pdf")
    assert chunks[0].text == "hello world again"


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        Chunker(chunk_size=5, overlap=5)
