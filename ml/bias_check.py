"""Slice-based fairness check for the answerability gate (Week 9: bias analysis).

A question-answering system should not be worse for some *kinds* of users just because of how
they phrase things. We have no demographic data, so we slice by phrasing instead:
question length and opening word ("what / why / how / who ..."). For each slice we compare the
error rates of out-of-fold predictions (every row is scored by a model that never saw it).

  false-negative rate  = in-PDF questions wrongly treated as NOT in the PDF  (user gets a "not from the PDF" warning
                         although the answer was there)
  false-positive rate  = off-topic questions wrongly treated as in the PDF  (user gets an unsupported answer
                         presented as sourced)

The headline number is the gap between the best and worst slice (the "equal opportunity" gap).

Usage:  python -m ml.bias_check
"""
import json

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import settings
from src.gate.features import FEATURE_NAMES
from src.gate.gate import AnswerabilityGate

MIN_SLICE = 8   # slices with fewer rows are too noisy to compare


def slice_report(frame: pd.DataFrame, column: str) -> list[dict]:
    rows = []
    for name, part in frame.groupby(column, observed=True):
        if len(part) < MIN_SLICE:
            continue
        positives, negatives = part[part.label == 1], part[part.label == 0]
        rows.append({
            "slice": str(name),
            "n": len(part),
            "accuracy": round(float((part.pred == part.label).mean()), 3),
            "false_negative_rate": None if positives.empty else round(float((positives.pred == 0).mean()), 3),
            "false_positive_rate": None if negatives.empty else round(float((negatives.pred == 1).mean()), 3),
        })
    return rows


def gap(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r[key] is not None]
    return round(max(values) - min(values), 3) if len(values) > 1 else None


def main():
    gate = AnswerabilityGate.load()
    if gate is None:
        raise SystemExit("No trained gate found. Run `python -m ml.train_gate` first.")
    frame = pd.read_csv(settings.gate_data_dir / "gate_dataset.csv").drop_duplicates(subset="question")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    frame["pred"] = cross_val_predict(clone(gate.model), frame[FEATURE_NAMES].to_numpy(), frame["label"].to_numpy(), cv=cv)

    frame["length_bucket"] = pd.cut(frame.question.str.split().str.len(), [0, 6, 9, 100], labels=["short (≤6 words)", "medium (7-9)", "long (10+)"])
    frame["opening_word"] = frame.question.str.split().str[0].str.lower()

    report = {
        "method": "5-fold out-of-fold predictions; slices with fewer than %d rows are skipped" % MIN_SLICE,
        "by_length": slice_report(frame, "length_bucket"),
        "by_opening_word": slice_report(frame, "opening_word"),
    }
    for key in ("by_length", "by_opening_word"):
        report[key + "_gaps"] = {m: gap(report[key], m) for m in ("false_negative_rate", "false_positive_rate")}

    out = settings.gate_dir / "bias_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for key in ("by_length", "by_opening_word"):
        print(f"\n{key}")
        print(pd.DataFrame(report[key]).to_string(index=False))
        print("gaps:", report[key + "_gaps"])
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
