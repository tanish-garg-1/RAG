"""Trains the answerability gate on data/gate/gate_dataset.csv and saves a versioned bundle.

Steps (the Week 3 data-prep checklist, in order):
  1. load + clean   2. stratified train/test split   3. cross-validate candidate models
  4. pick the best  5. evaluate once on the untouched test set
  6. compare with the old fixed-threshold rule   7. refit on all data, save model + metrics + SHAP plot

Usage:  python -m ml.train_gate
"""
import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")   # draw plots to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import settings
from src.gate.features import FEATURE_NAMES

SEED = 42


def load_clean(path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    before = len(frame)
    frame = frame.drop_duplicates(subset="question").dropna(subset=FEATURE_NAMES + ["label"])
    print(f"Loaded {before} rows, kept {len(frame)} after dropping duplicates / missing values.")
    print(frame.label.value_counts().rename({1: "in PDF", 0: "not in PDF"}).to_string())
    return frame


def score(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def main():
    frame = load_clean(settings.gate_data_dir / "gate_dataset.csv")
    X, y = frame[FEATURE_NAMES].to_numpy(), frame["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    candidates = {
        # Scaling matters for logistic regression (features live on very different scales: q_words ~ 8, scores 0..1).
        "logistic_regression": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.1, random_state=SEED),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results = {}
    for name, model in candidates.items():
        auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        cv_results[name] = {"roc_auc_mean": round(auc.mean(), 4), "roc_auc_std": round(auc.std(), 4), "f1_mean": round(f1.mean(), 4)}
        print(f"CV {name:20s} AUC {auc.mean():.3f} ± {auc.std():.3f}   F1 {f1.mean():.3f}")

    best_name = max(cv_results, key=lambda n: cv_results[n]["roc_auc_mean"])
    print(f"Selected: {best_name}")

    best = candidates[best_name].fit(X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]
    test_metrics = score(y_test, (proba >= 0.5).astype(int), proba)
    print("Held-out test:", test_metrics)
    print("Confusion matrix [[TN FP] [FN TP]]:", confusion_matrix(y_test, proba >= 0.5).tolist())

    # The old rule: in PDF when the best reranker score >= 0.6, no learning involved.
    top = X_test[:, FEATURE_NAMES.index("top_rerank")]
    baseline = score(y_test, (top >= settings.high_confidence).astype(int), top)
    print(f"Baseline (top_rerank >= {settings.high_confidence}):", baseline)

    # Deploy model: refit on ALL rows (more data helps with a small dataset); the numbers above stay the honest estimate.
    final = candidates[best_name].fit(X, y)

    out_dir = settings.gate_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    background = X[rng.choice(len(X), size=min(30, len(X)), replace=False)]
    joblib.dump(
        {"model": final, "feature_names": FEATURE_NAMES, "background": background, "version": settings.gate_version},
        settings.gate_model_path,
    )
    metrics = {
        "version": settings.gate_version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": best_name,
        "n_rows": int(len(frame)),
        "n_in_pdf": int(y.sum()),
        "cross_validation": cv_results,
        "test": test_metrics,
        "baseline_fixed_threshold": baseline,
        "confusion_matrix": confusion_matrix(y_test, proba >= 0.5).tolist(),
        "features": FEATURE_NAMES,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_shap_plot(final, X, out_dir / "shap_summary.png")
    print(f"Saved model, metrics.json and shap_summary.png to {out_dir}")


def save_shap_plot(model, X, path):
    import shap
    explainer = shap.Explainer(lambda data: model.predict_proba(data)[:, 1], X[: min(30, len(X))], feature_names=FEATURE_NAMES)
    values = explainer(X[: min(100, len(X))])
    plt.figure()
    shap.plots.bar(values, show=False)
    plt.title("Mean |SHAP|: which signals drive the in-PDF decision")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
