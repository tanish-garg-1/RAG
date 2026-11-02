import json
from pathlib import Path

import joblib
import numpy as np

from config import settings
from src.gate.features import FEATURE_DOCS, FEATURE_NAMES, extract_features
from src.models import RetrievedChunk


class AnswerabilityGate:
    """A trained classifier: P(the PDF answers this question) from retrieval signals.

    It replaces the two hand-picked score thresholds with a model learned from labelled
    questions (see ml/build_gate_dataset.py and ml/train_gate.py), and it can explain each
    decision with SHAP (explain()).
    """

    def __init__(self, model, background: np.ndarray, version: str, metrics: dict):
        self.model = model
        self.background = background      # a small sample of training rows: SHAP's "typical question"
        self.version = version
        self.metrics = metrics
        self._explainer = None

    @classmethod
    def load(cls, path: Path | None = None) -> "AnswerabilityGate | None":
        """Returns None when no trained gate exists, so the app falls back to the fixed thresholds."""
        path = path or settings.gate_model_path
        if not path.exists():
            return None
        bundle = joblib.load(path)
        if bundle["feature_names"] != FEATURE_NAMES:
            raise RuntimeError("The saved gate was trained on different features; retrain with ml/train_gate.py.")
        metrics_file = path.with_name("metrics.json")
        metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
        return cls(bundle["model"], bundle["background"], bundle["version"], metrics)

    def predict_proba(self, question: str, chunks: list[RetrievedChunk]) -> float:
        return self.predict_proba_matrix(np.array([extract_features(question, chunks)]))[0]

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def explain(self, question: str, chunks: list[RetrievedChunk]) -> dict:
        """SHAP breakdown of one decision, in probability units.

        base_value + sum(contributions) == probability. Positive contributions pushed the
        decision toward "answered by the PDF", negative ones toward "not in the PDF".
        """
        import shap   # imported lazily: it takes a few seconds and is only needed when explaining

        if self._explainer is None:
            # Model-agnostic explainer over predict_proba. With <= 10 features SHAP computes exact Shapley values.
            self._explainer = shap.Explainer(self.predict_proba_matrix, self.background, feature_names=FEATURE_NAMES)

        features = np.array([extract_features(question, chunks)])
        explanation = self._explainer(features)
        values = explanation.values[0]
        contributions = [
            {
                "feature": name,
                "description": FEATURE_DOCS[name],
                "value": float(features[0][i]),
                "shap": float(values[i]),
            }
            for i, name in enumerate(FEATURE_NAMES)
        ]
        contributions.sort(key=lambda c: abs(c["shap"]), reverse=True)
        return {
            "base_value": float(explanation.base_values[0]),
            "probability": float(self.predict_proba_matrix(features)[0]),
            "contributions": contributions,
        }
