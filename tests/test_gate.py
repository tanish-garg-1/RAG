import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.gate.features import FEATURE_NAMES, content_words, extract_features
from src.gate.gate import AnswerabilityGate
from src.models import Chunk, RetrievedChunk
from src.pipeline.scope_checker import ScopeChecker


def chunk(text="Monopolies create lasting value for founders.", rerank=0.9, dense=0.7, bm25=None, bm25_rank=None):
    return RetrievedChunk(Chunk("id", text, "doc.pdf", 3), dense_score=dense, bm25_score=bm25, bm25_rank=bm25_rank, rerank_score=rerank)


def test_feature_vector_matches_names_and_values():
    chunks = [chunk(rerank=0.9, bm25=4.0, bm25_rank=1), chunk("other", rerank=0.2, dense=0.4)]
    features = dict(zip(FEATURE_NAMES, extract_features("Why do monopolies create value?", chunks)))
    assert len(features) == len(FEATURE_NAMES)
    assert features["top_rerank"] == 0.9
    assert features["rerank_gap"] == pytest.approx(0.7)
    assert features["n_relevant"] == 2
    assert features["max_bm25"] == 4.0 and features["top_in_bm25"] == 1.0
    assert features["q_words"] == 5
    assert 0 < features["q_overlap"] <= 1


def test_no_chunks_gives_all_zero_features():
    assert extract_features("anything", []) == [0.0] * len(FEATURE_NAMES)


def test_content_words_drops_stopwords():
    assert content_words("What is the capital of Australia?") == {"capital", "australia"}


@pytest.fixture
def tiny_gate(tmp_path):
    """A gate trained on synthetic rows where 'in PDF' simply means a high top_rerank."""
    rng = np.random.default_rng(0)
    X = rng.random((200, len(FEATURE_NAMES)))
    y = (X[:, 0] > 0.5).astype(int)
    path = tmp_path / "gate.joblib"
    joblib.dump({"model": LogisticRegression().fit(X, y), "feature_names": FEATURE_NAMES, "background": X[:20], "version": "test"}, path)
    return AnswerabilityGate.load(path)


def test_load_returns_none_when_no_model_exists(tmp_path):
    assert AnswerabilityGate.load(tmp_path / "missing.joblib") is None


def test_gate_prefers_strong_matches(tiny_gate):
    strong = tiny_gate.predict_proba("monopolies create value", [chunk(rerank=0.99)])
    weak = tiny_gate.predict_proba("monopolies create value", [chunk(rerank=0.01)])
    assert strong > weak


def test_shap_contributions_add_up_to_the_prediction(tiny_gate):
    explanation = tiny_gate.explain("monopolies create value", [chunk(rerank=0.95)])
    total = explanation["base_value"] + sum(c["shap"] for c in explanation["contributions"])
    assert total == pytest.approx(explanation["probability"], abs=1e-6)
    assert explanation["contributions"][0]["feature"] == "top_rerank"   # the feature that defines the label


class NeverCalledLLM:
    def complete(self, *args, **kwargs):
        raise AssertionError("LLM should not be consulted for a confident gate decision")


def test_scope_checker_uses_gate_for_confident_decisions(tiny_gate):
    checker = ScopeChecker(NeverCalledLLM(), gate=tiny_gate)
    assert checker.check("monopolies create value", [chunk(rerank=0.99)]).method == "gate_high"
    low = checker.check("monopolies create value", [chunk(rerank=0.01)])
    assert low.method == "gate_low" and not low.in_pdf and low.gate_proba is not None
