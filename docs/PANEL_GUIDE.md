# Panel guide: understanding every part of this project

Read this top to bottom once, then use the Q&A bank at the end for practice. File names are real; open them while you read.

---

## 0. The 60-second version

**Problem.** People keep knowledge in PDFs. A plain chatbot cannot be trusted with them: it hallucinates, it does not say where an answer came from, and nobody can tell *why* it answered the way it did.

**Solution.** A PDF question-answering system that
1. answers **only from the document**, with page citations;
2. **decides honestly whether the document covers the question** (a trained classifier), and if not, still helps but shows a bold **NOT FROM THE PDF** warning;
3. can **explain that decision** (SHAP);
4. **survives provider failures** (Groq first, Azure OpenAI as automatic backup);
5. can answer in different **tones**, reads **figures and scanned pages** (Gemini), and is delivered as an **API + web UI + Docker** with **monitoring and tests**.

**What I built vs. what I used.** Libraries do the heavy lifting for the building blocks (PyMuPDF, FastEmbed, ChromaDB, rank-bm25, LangGraph, scikit-learn, SHAP, FastAPI, Streamlit). The design of the pipeline and the following parts are my own work: the chunking-with-page-citation scheme, the hybrid retrieval fusion, the corrective-retry graph, the **answerability gate** (dataset, features, training, evaluation, explanation), the **LLM fallback router**, the persona prompts, the vision cache/fallback, the API, UI, metrics and tests.

---

## 1. The life of one question (memorise this)

```
User types a question in Streamlit
  → POST /chat {question, persona, explain}                            src/api/main.py
  → RAGPipeline.answer()                                               src/pipeline/rag_pipeline.py
      LangGraph state machine                                          src/graph/build.py, nodes.py
      1. retrieve                                                      src/retrieval/retriever.py
           a. embed the question → 384 numbers                         src/indexing/embedder.py
           b. dense search in ChromaDB (cosine, top 20)                src/indexing/vector_store.py
           c. keyword search with BM25 (top 20)                        src/indexing/bm25_store.py
           d. fuse the two ranked lists with Reciprocal Rank Fusion    src/retrieval/hybrid.py
           e. cross-encoder re-scores the best 10, keep top 4          src/retrieval/reranker.py
      2. check_scope  (is the question covered by the PDF?)            src/pipeline/scope_checker.py
           features → trained gate → P(in PDF)                         src/gate/
           P ≥ 0.75 → yes, P ≤ 0.25 → no, in between → ask the LLM
      3a. in PDF       → generate_from_pdf  (answer ONLY from the 4 excerpts, cite pages)
      3b. weak match   → rewrite_query → back to step 1 (at most once)
      3c. not in PDF   → web_search (Tavily, only if TAVILY_API_KEY is set)    src/search/web_search.py
                         → generate_general (answer from the web results with [n] links,
                           or from model knowledge, flagged, if there is no key / search fails)
  → LLMRouter: Groq → (on any error) Azure OpenAI                      src/generation/llm_router.py
  → response: answer, sources, verdict, gate probability, SHAP, timings
  → logged to logs/queries.jsonl, counted in Prometheus metrics
```

Be able to draw this on a whiteboard. `python main.py graph` prints the LangGraph part as a Mermaid diagram.

---

## 2. Component by component

For each: **what it does → why → how it works → alternatives I rejected → what to say if asked.**

### 2.1 PDF loading (`src/ingestion/pdf_loader.py`)
- **What:** PyMuPDF opens the PDF and returns `(page_number, text)` pairs.
- **Why PyMuPDF:** fast, pure extraction, gives page numbers and can also extract images and render pages.
- **Detail:** a page with no extractable text is "scanned" (an image of text). For those, `iter_images` renders the whole page to PNG so a vision model can transcribe it. Embedded images smaller than 150 px are skipped (icons, bullets).

### 2.2 Chunking (`src/ingestion/chunker.py`)
- **What:** splits each page into windows of **200 words** that overlap by **40 words**; the step is 160.
- **Why chunk at all:** an embedding represents *one meaning*. If you embed a whole page, many topics blur into one vector and retrieval gets vague. If you cut too small, a chunk loses the context needed to answer. 200 words (~1-2 paragraphs) is a common middle ground.
- **Why overlap:** a fact sitting on a boundary would otherwise be cut in half. The overlap repeats the last 40 words so it appears whole in at least one chunk.
- **Why chunks never cross pages:** every chunk keeps an exact page number, which is what makes **citations correct**.
- **Words, not tokens:** simple and model-independent; tokenisers differ between models.
- **Edge cases handled:** pages with fewer than 2 words (a lone page number) are skipped; `overlap >= chunk_size` raises an error because the window would never advance.

### 2.3 Embeddings (`src/indexing/embedder.py`)
- **What:** `BAAI/bge-small-en-v1.5` via FastEmbed (ONNX runtime, runs on CPU) turns text into a list of **384 numbers**.
- **Intuition:** the model is trained so that texts with similar meaning get vectors pointing in similar directions. "How do monopolies make money?" and "Why do monopolists earn profits?" end up close together even though they share few words.
- **Similarity measure:** **cosine similarity**, the cosine of the angle between two vectors (1 = same direction, 0 = unrelated).
- **Query vs. passage:** bge models expect questions to carry an instruction prefix; `query_embed` adds it, `passage_embed` is used for chunks. Using the right one matters for accuracy.
- **Why a local model:** free, no rate limits, no document text leaves the machine for embedding, and it is fast enough.

### 2.4 Vector database (`src/indexing/vector_store.py`)
- **What:** ChromaDB, persisted to `data/vector_store/`. It stores each chunk's vector, text and metadata (source, page, kind).
- **How search works:** Chroma builds an **HNSW** graph index, so it finds approximate nearest neighbours without comparing against every vector. We ask for cosine distance and convert: `similarity = 1 - distance`.
- **Re-ingest safety:** `delete_source` removes a file's old chunks before adding new ones, so ingesting twice never duplicates.
- **Why Chroma:** embedded (no server to run), persistent, simple Python API. Alternatives (Pinecone, Weaviate, pgvector) add infrastructure we do not need at this scale.

### 2.5 Keyword search, BM25 (`src/indexing/bm25_store.py`)
- **Why in addition to embeddings:** embeddings are great for *meaning* but can miss *exact* terms: names, codes, rare words ("Xanadu"). BM25 is the classic keyword ranking algorithm (a refined TF-IDF): it rewards chunks that contain the query's rare words many times and penalises very long chunks.
- **Stop words** are removed so "what is the" does not match everything.
- **Derived data:** the BM25 index is always rebuilt from the chunks in Chroma, so the two indexes cannot drift apart.

### 2.6 Reciprocal Rank Fusion (`src/retrieval/hybrid.py`)
- **Problem:** cosine scores (0..1) and BM25 scores (unbounded) are on different scales; you cannot just add them.
- **Solution:** use only the **ranks**. Each list gives a chunk `1 / (60 + rank)`; a chunk found by both lists gets both contributions. Chunks both searches like rise to the top.
- **Why k = 60:** a standard smoothing constant from the original RRF paper; it stops rank 1 from dominating completely.
- **Important design rule:** the RRF score is only used for ordering. It is **never** used for the in-PDF decision because it encodes rank, not relevance.

### 2.7 Cross-encoder reranker (`src/retrieval/reranker.py`)
- **Bi-encoder vs. cross-encoder:** the embedding model (bi-encoder) encodes the question and each chunk *separately*, so it is fast but approximate. A cross-encoder reads the question **and** the chunk **together** and outputs one relevance score, so it is far more accurate but slower, so we run it only on the best 10 candidates.
- **Score:** the model outputs a raw number (logit); a sigmoid squashes it to 0..1. This is the most trustworthy relevance signal in the system and the main input to the gate.
- **Pipeline shape:** wide-and-cheap first (dense + BM25 → 20+20 → fuse), narrow-and-expensive last (rerank 10 → keep 4). This "retrieve then rerank" pattern is standard in production search.

### 2.8 The LangGraph workflow (`src/graph/`)
- **What:** the pipeline is a **state machine**. Nodes are steps (`retrieve`, `check_scope`, `rewrite_query`, `generate_from_pdf`, `generate_general`); a shared `RAGState` dictionary flows through them; a **conditional edge** (`route_after_scope`) chooses the next step.
- **Corrective retry:** if the question looks out of scope but the best score is still above 0.005 (some relevance), the LLM rewrites the question into a better search query and the graph loops back to `retrieve`, at most **once** (`max_rewrites`). Near-zero scores skip the retry because rewording cannot find a topic that is not in the document.
- **Key safety rule:** the scope check always judges the **original** question, never the rewrite, so a rewrite can improve the search but cannot change what was asked. There is a test for exactly this.
- **Why LangGraph and not a plain function:** explicit, testable routing; the path taken (`retrieve → check_scope → rewrite_query → …`) is recorded and shown to the user.

### 2.9 The answerability gate (`src/gate/`, `ml/`), the main ML contribution
**The question it answers:** "does the PDF really contain the answer to this question?"

**Why not a fixed threshold:** the first version said "reranker score ≥ 0.6 → yes, < 0.05 → no, else ask the LLM". Two hand-picked numbers ignore the other available signals and are never validated. A model learns the boundary from labelled examples and can be measured against the old rule.

**Features** (`src/gate/features.py`): nine numbers computed from the retrieval result.
| Feature | Meaning |
|---|---|
| `top_rerank` | cross-encoder score of the best chunk |
| `rerank_gap` | best minus second-best: one clear winner vs. a crowd of mediocre chunks |
| `mean_rerank` | average score of the 4 retrieved chunks |
| `n_relevant` | how many chunks scored above 0.1 |
| `top_dense` | embedding similarity of the best chunk |
| `max_bm25` | strongest exact-keyword match |
| `top_in_bm25` | 1 if the best chunk was also found by keyword search (two independent methods agree) |
| `q_words` | question length |
| `q_overlap` | share of the question's content words that appear in the best chunk |

**Dataset** (`ml/build_gate_dataset.py`): labels come from *how the questions were made*, not from the system's own opinion (that would be circular):
- label **1** (in the PDF): an LLM reads a random passage and writes a question that the passage answers. Questions about layout ("which page…") are filtered out.
- label **0** (not in the PDF): 88 general-knowledge questions (`ml/out_of_doc_questions.txt`) plus 25 **hard negatives** (`ml/hard_negatives.txt`): startup/business questions that *sound* on-topic but the book does not cover ("What is a SAFE note?"). Every negative can carry a key term, and it is **dropped if that term appears anywhere in the book**, so the label is verifiable.
- For each question we run the *real* retriever and record the nine features.

**Training** (`ml/train_gate.py`): the Week 3 checklist, in order
1. load and clean (drop duplicates, missing values);
2. **stratified** train/test split, 75/25 (stratified = both sets keep the same class proportions);
3. **5-fold cross-validation** on the training part for two candidates: logistic regression **with feature scaling** (`StandardScaler`, because `q_words` is ~8 while scores are 0..1) and gradient boosting;
4. choose the best by cross-validated ROC-AUC;
5. evaluate **once** on the untouched test set (precision, recall, F1, ROC-AUC, confusion matrix);
6. compare with the **old fixed-threshold rule** on the same test rows;
7. refit on all the data for deployment (small dataset: more data helps) and save a versioned bundle to `models/gate/v1/` with `metrics.json`.


**Results (real numbers from `models/gate/v1/metrics.json`; 182 questions: 69 in the PDF, 113 not)**

| | Gate (logistic regression) | Old fixed rule (`top_rerank ≥ 0.6`) |
|---|---|---|
| 5-fold CV ROC-AUC (train part) | 0.984 ± 0.033 | n/a (no training) |
| Held-out test accuracy (46 questions) | **97.8 %** | **97.8 %** |
| Test precision / recall / F1 | 1.00 / 0.94 / 0.97 | 1.00 / 0.94 / 0.97 |
| Test ROC-AUC | 0.957 | 0.998 |
| Test confusion matrix (TN FP / FN TP) | 29 0 / 1 16 | same |

**Say this plainly, because a good panel will ask: the gate does NOT beat the old rule on accuracy.** On this book, one feature (`top_rerank`) already separates the classes almost perfectly, and the test set is small (46 questions, so one question is 2 percentage points). What the gate adds is (1) a calibrated *probability* instead of a hard cut, (2) an explanation of every decision, (3) a measured, versioned, retrainable component, and (4) room to improve on harder documents. Selected model: logistic regression (CV AUC 0.984) over gradient boosting (0.971): simpler, equally good, and more stable.

**A lesson from building it (good to tell):** the first dataset flagged "Who discovered penicillin?" as *not in the PDF* and the model "got it wrong". Searching the book showed it *does* mention penicillin (also machine learning and the Berlin Wall). The **labels** were wrong, not the model. That is why every negative is now checked against the book text.

**Bias check findings** (`models/gate/v1/bias_report.json`, out-of-fold predictions):
- By length: short questions (≤ 6 words) have a **50 % false-negative rate** versus 0-2 % for longer ones. Caveat: only about 4 short in-PDF questions exist, so this is 2 errors; it is a flag to investigate with more data, not a proven effect. Likely cause: short questions give the cross-encoder little to match.
- By opening word: "what" questions have an 8.7 % false-negative rate; "how" and "why" are 0 %. The slices are small (n = 9-76).
- Mitigation if pursued: collect more short in-PDF questions, add a "short question" augmentation, or lower the decision threshold for short questions.

**At runtime** (`src/pipeline/scope_checker.py`): probability ≥ 0.75 → in the PDF; ≤ 0.25 → not in the PDF; in between → the old LLM "do these excerpts answer the question?" check. So the cheap model decides the clear cases and the expensive LLM only the borderline ones. If no gate file exists, the app silently uses the old thresholds.

**Explainability with SHAP** (`src/gate/gate.py::explain`)
- **Idea (Shapley values from game theory):** treat the nine features as players in a team that produces the prediction. A feature's SHAP value is its average marginal contribution over all possible orderings in which features could be "added".
- **Properties to quote:** contributions **add up exactly**: `base value + Σ SHAP = final probability` (there is a unit test for this). Positive = pushes toward "in the PDF", negative = away.
- **Base value:** the average prediction over a small background sample of training questions ("a typical question").
- **Exact, not approximate:** with ≤ 10 features SHAP enumerates all coalitions, so values are exact. `explain` uses the model-agnostic explainer over `predict_proba`, so it works for either model type.
- **Where you see it:** the "How was this decided?" expander (bar chart), and `models/gate/v1/shap_summary.png` (global importance).

**Bias check** (`ml/bias_check.py`): no demographic data exists for questions, so slices are *phrasing* groups (question length, opening word). It reports false-negative and false-positive rates per slice using out-of-fold predictions, and the gap between the best and worst slice.

**Limitations to state yourself, before they are asked:**
- Small dataset (182 questions) from **one book**; results are an estimate, not a guarantee.
- In-PDF questions are **LLM-written**, so the gate may partly learn "LLM-style phrasing"; out-of-PDF questions are generic, so near-topic-but-uncovered questions are under-represented (those are the hard cases).
- The gate must be retrained when the document type changes.

### 2.10 The LLM router (`src/generation/llm_router.py`)
- **What:** tries Groq; on *any* exception (rate limit 429, timeout, outage, bad key) it sends the same request to Azure OpenAI. If both fail, it raises one error that lists both reasons.
- **Why lazy clients:** providers are created on first use, so ingestion works with no key and an unconfigured Azure simply never participates.
- **Why one interface:** the router has the same `complete()`/`last_usage` as a single client, so the graph nodes and scope checker do not know there is more than one provider.
- **Counters:** calls, fallbacks, failures, per-provider ok/error, exposed in `/stats` and `/metrics`, and `provider` is stored with every answer.
- **Why Groq first:** free tier and very fast. **Why Azure second:** enterprise-grade, high-availability, and the course covers Azure AI. The fallback is about **availability**, which is the production concern.

### 2.10b Web search with Tavily (`src/search/web_search.py`, `web_search` node in `src/graph/`)
- **What:** for out-of-PDF questions only, the question is sent to the Tavily search API; the top 4 snippets go to the LLM with a prompt that says "answer ONLY from these numbered results and cite [n]". The UI shows a blue banner and a Web sources list.
- **Three answer sources, always labelled:** `pdf` (green, page citations), `web` (blue, link citations), `model` (yellow, general knowledge). `answer_source` is in the API response, the log and `/stats`.
- **Real-time questions** (`needs_fresh_results`: today, latest, news, price, score, weather, "current president", years 2025+) always go to the web, even if the PDF overlaps the topic, because a PDF cannot know today's facts. They search Tavily's **news** mode for the last 30 days (topped up with a normal search if fewer than 3 hits), and the prompt includes today's date plus each result's publication date so the model can say how fresh the answer is. Questions about the PDF itself ("the book", "page 5") are never sent to the web (searching them returned unrelated books).
- **Why this matters (a real example from my testing):** a plain search for "most recent World Cup winner" returned old list pages and an out-of-date answer; the news mode returned October 2026 pages and a current answer.
- **Privacy:** only the question text leaves the machine, never PDF text, and the PDF path never calls Tavily at all (tested).
- **Prompt injection:** web pages can contain text like "ignore previous instructions". The prompt marks results as untrusted data; I tested a poisoned result and the model ignored it. This reduces the risk but cannot eliminate it, so say that.
- **Graceful degradation:** any search failure (timeout, 401, 429, no results) falls back to model knowledge and shows why; the question is still answered.
- **Optional by design:** the `web_search` node is only added to the graph when a key is set, so the default behaviour and all PDF answers are unchanged.
- **Honest limits:** search snippets can be wrong or biased, so the answer is only as good as the sources; I tested with a local fake Tavily server, so try it once with a real key before the demo.

### 2.11 Personas / tone (`src/generation/prompts.py`)
- Five styles: default, formal, friendly, concise, ELI5. Each adds a **style paragraph + one example of the voice** to the system prompt (*few-shot prompting*: models imitate an example more reliably than they obey an adjective).
- **Safety design:** the grounding and citation rules come first and the style block says "changes only the wording, never the rules above, never the citations". Tests assert the rules are always preserved.

### 2.12 Vision with Gemini (`src/vision/`)
- At ingest time, figures/charts are described and scanned pages are transcribed, then stored as normal chunks marked `kind="image"`, so questions about a chart can be answered and cited as `(figure, p. N)`.
- **Provider order:** Gemini (cloud) → Ollama (local) when Gemini's quota is exhausted; the first provider is retried after a 60 s cooldown.
- **Cache:** results are stored by SHA-256 of the image bytes, so re-ingesting never pays twice. Decorative images (logos) are detected and dropped.
- **Honest caveat:** numbers read from charts are model-generated and may be slightly off, which the answer prompt tells the LLM to say.

### 2.13 API (`src/api/`)
- **FastAPI** with typed request/response models (`schemas.py`, Pydantic): invalid input (unknown persona, empty question) is rejected with HTTP 422 automatically.
- Status codes carry meaning: 409 empty index, 400 non-PDF upload, 413 too large, 422 unreadable PDF, 502 all LLM providers failed.
- A lock serialises requests because the pipeline holds shared state (Chroma, BM25, graph).
- Models are warmed up in a background thread at startup so the first user question is not slow.
- `create_app(pipeline=...)` accepts a fake pipeline: that is how the API is tested without models or keys.

### 2.14 UI (`app/streamlit_app.py`)
- Contains **no RAG logic**: it only calls the API. This separation means the backend can be used by any client (and tested alone).
- Chat tab (persona, sources, decision explanation), Monitoring tab (from `/stats`), About tab.

### 2.15 Monitoring (`src/monitoring/metrics.py`, `logs/queries.jsonl`, `/stats`, `/metrics`)
- **Two views of the same events:** a JSON-lines log (one record per question: verdict, method, gate probability, provider, persona, scores, timings, token usage) and Prometheus counters/histograms (`rag_questions_total{verdict}`, `rag_llm_answers_total{provider}`, `rag_llm_fallbacks_total`, `rag_answer_seconds`).
- **What you would watch in production:** share of out-of-PDF questions (are users asking things the document lacks?), fallback rate (is Groq unhealthy?), p95 latency, errors.
- A private Prometheus registry is used so tests can create many apps without "duplicate metric" errors.

### 2.16 Docker and CI
- `Dockerfile`: dependencies are installed *before* copying code, so editing code does not reinstall packages (layer caching). `docker-compose.yml` runs two containers from one image (API and UI), mounts `data/` and `logs/` as volumes so the index survives restarts, and the UI waits for the API's healthcheck.
- `.github/workflows/ci.yml`: on every push, install dependencies and run the tests, then build the Docker image. Tests use fakes, so CI needs no secrets.

### 2.17 Tests (`tests/`)
- 57 tests (run `pytest`): chunker, RRF, scope checker, graph routing (including the retry cap and "judge the original question"), vision fallback, **router fallback**, **persona safety**, **gate features / SHAP additivity / gate-driven decisions**, **API contract**.
- **Fakes instead of mocks of the network:** small stand-in classes (`FakeClient`, `FakePipeline`) keep tests fast, deterministic and key-free.

---

## 3. Mapping to the course syllabus

| Week | Where to point |
|---|---|
| 1 Python | typed, modular package; dataclasses (`src/models.py`), generators (`iter_images`), context managers |
| 2 AI/ML intro | this is *narrow AI*: retrieval + a generative model + a classical ML classifier, each used for what it is good at |
| 3 Data handling | the gate dataset: collection, cleaning, feature engineering, scaling, stratified split, cross-validation |
| 4-6 GenAI/LLMs | transformers (embedding model, cross-encoder, Llama/GPT-OSS answer model), prompt engineering (grounding, citations, personas, few-shot), RAG, vector DB, API, Groq + Gemini |
| 7-8 Azure | Azure OpenAI as a backup provider (`azure_client.py`); the same code works against an Azure AI Foundry deployment by changing the endpoint and deployment name |
| 9 Ethics/XAI | SHAP explanations, slice-based bias check, transparency about out-of-document answers, limitations stated |
| 10-11 MLOps | versioned model artifact + metrics, Docker, logging, Prometheus metrics, CI |
| 12 Capstone | this repository |

---

## 4. Anticipated questions and strong answers

**Concepts**
0. *Does it search the web?* Only for questions the PDF cannot answer, and only if a Tavily key is set. Those answers cite web links and carry a blue "not from the PDF" banner; PDF answers never use the web. Without a key it falls back to the model's own knowledge, flagged in yellow.
1. *What is RAG and why use it?* Retrieval-Augmented Generation: fetch relevant passages first, then let the LLM answer from them. It reduces hallucination, lets the model use private/new data without retraining, and enables citations.
2. *Why not fine-tune the LLM on the PDF?* Fine-tuning is costly, bakes facts into weights (hard to update or cite), and still hallucinates. RAG updates by re-indexing and can point to the source page.
3. *What is an embedding?* A vector (here 384 numbers) representing the meaning of a text, so semantically similar texts are close in vector space.
4. *What is cosine similarity and why not Euclidean distance?* Cosine compares direction, ignoring length; for normalised text embeddings it is the standard measure.
5. *Why hybrid retrieval?* Dense search handles paraphrase, BM25 handles exact terms; each fails where the other works. RRF merges them without comparing incompatible scores.
6. *Why rerank?* Bi-encoders are fast but approximate; a cross-encoder sees question and chunk together and is much more accurate, so use it on a short list.
7. *What are the risks of chunking?* Too big: blurred meaning; too small: lost context; boundary splits, hence overlap.
8. *What is hallucination and how do you limit it here?* Confident wrong output. Limits: answer only from retrieved excerpts, page citations, an explicit "not from the PDF" label, low temperature (0.2), and a prompt rule to say what is missing.
9. *What does temperature do?* It scales randomness in token sampling; low = more deterministic and faithful.
10. *What is a transformer, briefly?* A neural architecture based on self-attention, which lets every token weigh every other token; it underlies embeddings, rerankers and LLMs.

**The ML gate**
11. *Why a classifier instead of thresholds?* It combines nine signals, is validated on held-out data and compared with the old rule, and can be explained and versioned.
12. *Where do the labels come from? Isn't that circular?* No: positive questions are generated *from a passage* (so the answer exists by construction); negatives are unrelated general-knowledge questions. The system's own verdicts are never used as labels.
13. *How do you know it isn't overfitting?* 5-fold cross-validation for model selection (AUC 0.984 ± 0.033), a separate untouched test set for the reported numbers (97.8 % accuracy), and a simple model class with nine features. Honest caveat: the test set has only 46 questions.
13b. *Does the gate beat the old threshold?* No, not on accuracy: both score 97.8 % on the test set. The gate adds probabilities, explanations and a retrainable, versioned component; on a harder or different document it could pull ahead, and I measured rather than assumed.
14. *Why ROC-AUC for model selection?* It measures ranking quality independent of any threshold; we then choose thresholds separately (0.25/0.75).
15. *Why scale features?* Logistic regression is sensitive to feature scale and its regularisation penalises large-valued features unfairly; `q_words` would dominate otherwise.
16. *What is precision vs. recall here?* Recall = of the questions the PDF answers, how many we correctly route to the PDF. Precision = of the ones we call "in the PDF", how many truly are. A false positive is worse: it presents an unsupported answer as sourced.
17. *What is SHAP?* A method assigning each feature its fair share of a prediction, based on Shapley values; the shares sum to the prediction minus the baseline.
18. *Why SHAP and not LIME?* SHAP has consistency guarantees and exact computation for few features; LIME fits a local approximation that can vary run to run.
19. *What bias did you look for, and what did you find?* Whether error rates differ by phrasing (length, opening word). Short questions (≤ 6 words) were wrongly routed away from the PDF in 50 % of in-PDF cases versus 0-2 % for longer ones, but that rests on about 4 short in-PDF questions, so it is a flag, not a conclusion. Fix: more short-question data.
20. *What if the document is a different type?* Retrain: rebuild the dataset from the new corpus (`ml.build_gate_dataset`) and run `ml.train_gate`. The model is versioned, so the old one stays available.

**Engineering**
21. *What happens when Groq is down?* The router falls back to Azure; the answer's `provider` field and the fallback counter show it. If Azure is not configured or also fails, the API returns 502 with both reasons.
22. *Is it safe to upload any file?* Only `.pdf` names are accepted, filenames are reduced to their base name (no path traversal), size is limited to 50 MB. Limitations: no virus scan, no auth.
23. *Why FastAPI + Streamlit separately?* Separation of concerns: the API is reusable and testable; the UI is a thin client.
24. *How would you scale it?* Move Chroma to a server/managed vector DB, run several API workers behind a load balancer (the lock and local files are the current single-instance limits), add a cache and queue for ingestion, add authentication and per-user indexes.
25. *How do you test code that calls external LLMs?* Behind a tiny interface (`complete()`), so tests inject fakes. Live behaviour is checked manually and by the evaluation questions.
26. *What does the Dockerfile do and why this order?* Install dependencies first so Docker caches that layer; code changes then rebuild in seconds.
27. *How is the model versioned?* `models/gate/<version>/` contains the artifact, `metrics.json` (with training time, data size, metrics) and plots; `GATE_VERSION` selects which one is loaded.
28. *What do you log, and why?* Per question: verdict, deciding method, scores, provider, persona, timings, token usage, for debugging, cost tracking and drift detection.
29. *Security/privacy?* Secrets only in `.env` (git-ignored); embeddings are computed locally; only the question and the 4 retrieved excerpts go to the LLM provider. Limitation: those excerpts do leave the machine.
30. *What would you improve next?* More and harder gate data (near-topic unanswerable questions), multi-document filtering, answer-faithfulness evaluation (e.g. Ragas), authentication, streaming responses.

**Honesty questions**
31. *What doesn't work well?* Chart numbers from the vision model can be off; very short or ambiguous questions; the gate was trained on one book; free-tier rate limits make answers slow at times.
32. *Which parts did you not write yourself?* The libraries named in section 0. Be specific about what you configured and combined.

---

## 5. Demo script (5 minutes)

1. `docker compose up` (or the two commands in the README). Open the UI. Point at the sidebar: index size, gate loaded, Groq ✅.
2. Ask a question the PDF answers. Show: green banner, page citations, **Sources**, then **How was this decided?** → the SHAP chart; say which feature mattered most.
3. Ask an unrelated question (e.g. "What is the capital of Australia?"). Show the yellow **NOT FROM THE PDF** banner and the explanation of why.
4. Change the style to **Explain like I'm 5**, ask the first question again.
5. Open the **Monitoring** tab: verdict counts, providers, latency.
6. (Optional, if Azure is configured) break the Groq key in `.env`, restart the API, ask again: the LLM metric shows `azure`; the fallback counter increments.
7. Close with `/docs` (Swagger) and `pytest` going green.

**Have a backup:** record the demo as a short video in case the network fails.
