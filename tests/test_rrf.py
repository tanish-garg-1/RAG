import pytest

from src.models import Chunk
from src.retrieval.hybrid import reciprocal_rank_fusion


def chunk(cid):
    return Chunk(chunk_id=cid, text=cid, source="doc.pdf", page=1)


def test_chunk_found_by_both_searches_ranks_first():
    dense = [(chunk("a"), 0.9), (chunk("b"), 0.8)]
    bm25 = [(chunk("c"), 12.0), (chunk("b"), 9.0)]
    fused = reciprocal_rank_fusion(dense, bm25, k=60)
    assert fused[0].chunk.chunk_id == "b"
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 62)


def test_raw_scores_and_ranks_are_preserved():
    fused = reciprocal_rank_fusion([(chunk("a"), 0.75)], [(chunk("a"), 4.2)], k=60)
    rc = fused[0]
    assert (rc.dense_score, rc.dense_rank, rc.bm25_score, rc.bm25_rank) == (0.75, 1, 4.2, 1)


def test_chunk_missing_from_one_list_keeps_none_for_that_side():
    fused = {rc.chunk.chunk_id: rc for rc in reciprocal_rank_fusion([(chunk("a"), 0.5)], [(chunk("b"), 3.0)])}
    assert fused["a"].bm25_score is None and fused["a"].bm25_rank is None
    assert fused["b"].dense_score is None and fused["b"].dense_rank is None


def test_only_rank_matters_not_score_scale():
    # A huge BM25 score must not outweigh a dense hit at the same rank.
    fused = reciprocal_rank_fusion([(chunk("a"), 0.01)], [(chunk("b"), 9999.0)], k=60)
    assert fused[0].rrf_score == pytest.approx(fused[1].rrf_score)


def test_empty_inputs():
    assert reciprocal_rank_fusion([], []) == []
