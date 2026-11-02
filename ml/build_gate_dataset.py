"""Builds the labelled dataset for the answerability gate.

  label 1  question the PDF DOES answer   (an LLM writes a question from a random passage)
  label 0  question the PDF does NOT answer (ml/out_of_doc_questions.txt: general-knowledge questions,
           ml/hard_negatives.txt: startup/business questions the book does not cover, verified absent from the text)

For every question we run the real retriever (no answer generation) and record the retrieval
features in src/gate/features.py. The labels come from how the questions were made, NOT from the
system's own verdict, so the model learns something the fixed thresholds could not.

Usage:  python -m ml.build_gate_dataset [--n-in 80]
Needs an indexed PDF (python main.py ingest) and, for the first run, a Groq key to write questions.
"""
import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd

from config import settings
from src.gate.features import FEATURE_NAMES, extract_features
from src.pipeline.rag_pipeline import RAGPipeline

QUESTION_PROMPT = (
    "You write exam questions. Read the passage and write ONE specific question that this passage answers.\n"
    "Rules: the question must make sense on its own, must not mention 'the passage', 'the text' or 'the author', "
    "and should be phrased the way a curious reader would ask it. Reply with only the question."
)


# Questions about the book's layout rather than its content ("which page mentions X?") are not real content questions.
LAYOUT_WORDS = ("page", "index", "chapter", "passage", "excerpt", "the text", "the author")


def is_good_question(question: str) -> bool:
    lowered = question.lower()
    return question.endswith("?") and len(question.split()) >= 4 and not any(w in lowered for w in LAYOUT_WORDS)


def generate_in_doc_questions(pipeline: RAGPipeline, n: int, cache: Path, seed: int) -> list[dict]:
    cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else []
    if len(cached) >= n:
        return cached[:n]

    done = {q["chunk_id"] for q in cached}
    chunks = [c for c in pipeline.vector_store.all_chunks() if c.kind == "text" and len(c.text.split()) >= 80]
    random.Random(seed).shuffle(chunks)

    for chunk in chunks:
        if len(cached) >= n:
            break
        if chunk.chunk_id in done:
            continue
        for attempt in range(4):   # free-tier rate limits: wait and retry
            try:
                question = pipeline.llm.complete(QUESTION_PROMPT, chunk.text, temperature=0.4).strip().strip('"')
                break
            except Exception as exc:
                print(f"  LLM error ({type(exc).__name__}), waiting 20s...")
                time.sleep(20)
        else:
            print("  giving up on this chunk")
            continue
        if is_good_question(question):
            cached.append({"question": question, "label": 1, "chunk_id": chunk.chunk_id, "page": chunk.page})
            cache.write_text(json.dumps(cached, indent=1), encoding="utf-8")
            print(f"  [{len(cached)}/{n}] p.{chunk.page}: {question}")
    return cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-in", type=int, default=80, help="number of in-PDF questions to generate")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    settings.gate_data_dir.mkdir(parents=True, exist_ok=True)
    pipeline = RAGPipeline()
    if pipeline.vector_store.count() == 0:
        raise SystemExit("The index is empty. Run `python main.py ingest` first.")

    print("Writing in-PDF questions...")
    in_doc = generate_in_doc_questions(pipeline, args.n_in, settings.gate_data_dir / "in_doc_questions.json", args.seed)
    in_doc = [q for q in in_doc if is_good_question(q["question"])]   # also cleans questions cached by older runs
    # Negatives are verified: a line may carry "| key term", and is dropped if that term appears in the book
    # (the first version wrongly labelled "Who discovered penicillin?" as not-in-PDF; the book mentions it).
    book_text = " ".join(c.text for c in pipeline.vector_store.all_chunks()).lower()
    out_doc = []
    for name in ("out_of_doc_questions.txt", "hard_negatives.txt"):
        for line in Path(__file__).with_name(name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            question, _, key_term = (part.strip() for part in line.partition("|"))
            if not key_term or key_term.lower() not in book_text:
                out_doc.append({"question": question, "label": 0})

    print(f"Retrieving for {len(in_doc)} in-PDF and {len(out_doc)} out-of-PDF questions...")
    rows = []
    for item in in_doc + out_doc:
        chunks = pipeline.retriever.retrieve(item["question"])
        rows.append({"question": item["question"], "label": item["label"], **dict(zip(FEATURE_NAMES, extract_features(item["question"], chunks)))})

    frame = pd.DataFrame(rows)
    out = settings.gate_data_dir / "gate_dataset.csv"
    frame.to_csv(out, index=False)
    print(f"Saved {len(frame)} rows to {out}\n{frame.label.value_counts().to_string()}")


if __name__ == "__main__":
    main()
