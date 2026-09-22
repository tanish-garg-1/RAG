# PDF RAG — hybrid retrieval in the terminal

Ask questions about your PDFs from the command line. Answers come from the PDF with page citations.
When a question isn't covered by the PDF, the app still answers from the LLM's general knowledge,
but flags it clearly: **⚠ NOT FROM THE PDF**. The analytics panel shows the evidence behind every verdict.

## How it works

```
question ─┬─ dense search (FastEmbed + ChromaDB) ─┐
          └─ keyword search (BM25) ───────────────┴─ RRF fusion ─ cross-encoder rerank ─ scope check ─ Groq
```

1. **Hybrid retrieval.** Vector search finds paraphrases, and BM25 finds exact terms (names, codes, rare words).
   Reciprocal Rank Fusion merges the two lists by rank, so their different score scales never need to be compared.
2. **Reranking.** A local cross-encoder reads the question and each candidate together and gives it a 0–1 relevance score.
3. **Scope check.** Uses the top rerank score (never the RRF score, which only reflects rank):
   - `>= 0.6`: in the PDF
   - `< 0.05`: out of the PDF
   - in between: one quick Groq call asks whether the excerpts actually answer the question
4. **Answer.** In-PDF answers use only the excerpts and cite pages. Out-of-PDF answers use general knowledge and carry the banner.

Everything except the Groq call runs locally. The models download on first use (~250 MB) to `data/models/`.

## Setup

```bash
python -m venv rag_env
rag_env\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then add your GROQ_API_KEY
```

## Usage

```bash
python main.py ingest                     # index every PDF in data/pdfs/
python main.py ingest path\to\file.pdf    # or a single file
python main.py ask "What was Xanadu?"     # one question, with analytics
python main.py ask "..." --no-scores      # answer only
python main.py chat                       # interactive loop: /scores /sources /quit
python main.py stats                      # index contents + in/out-of-PDF query summary
python main.py reset                      # clear the index
```

Re-ingesting a file replaces its chunks instead of duplicating them.

## Analytics panel

Each answer shows the verdict, how it was decided (`high_score`, `low_score`, `llm_check`),
and a table of the retrieved chunks with their rerank, dense, BM25 and RRF scores and ranks.
A `-` in the Dense or BM25 column means that search missed the chunk, which shows what hybrid adds.
Every query is also logged to `logs/queries.jsonl`.

## Tuning

All settings are in `config.py`: models, chunk size and overlap, candidate counts, and thresholds.
To recalibrate the thresholds for your documents, ask a handful of questions you know are in the PDF
and a handful you know are not, then read `top_rerank_score` in `logs/queries.jsonl` and set
`high_confidence` / `low_confidence` to separate the two groups.

On the sample book, in-PDF questions scored 0.375–1.0 and unrelated questions scored ≤ 0.001.

## Tests

```bash
pytest
```

## Project layout

```
main.py                     CLI (Typer)
config.py                   all settings
src/ingestion/              PDF text extraction and chunking
src/indexing/               embeddings, ChromaDB, BM25
src/retrieval/              RRF fusion, reranker, retriever
src/generation/             Groq client and prompts
src/pipeline/               scope checker and end-to-end pipeline
src/analytics/              terminal display and query log
tests/                      chunker, RRF and scope-checker tests
```
