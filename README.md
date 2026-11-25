# Sephra 💄

**Hybrid RAG beauty-product discovery.** Ask in plain English, for example
*"moisturizer for dry skin under $30, not greasy"*. Sephra enforces your
explicit requirements exactly, finds suitable products with semantic and
keyword search, and backs each one with customer reviews from that same
product, both positive and critical.

> Prices, ratings, availability and reviews come from a static public
> dataset snapshot ([Kaggle: Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews))
> and may not reflect current information. Sephra is **not affiliated with
> Sephora** and is not a shop: there is no cart, checkout or purchasing.

## Status

| Component | Status |
|---|---|
| Data cleaning, documents, chunking, embeddings, ChromaDB | ✅ done (original pipeline, unchanged) |
| Hard filters (price, rating, category, brand, stock, attributes, ingredients) | ✅ |
| Hybrid retrieval (dense + BM25 + RRF), parent documents, variant grouping | ✅ |
| Product-specific, balanced review evidence | ✅ |
| Reranker (optional, benchmarked, off by default) | ✅ |
| Rule-based query parser (no-API fallback) | ✅ |
| LLM query parser (ERNIE 5.5 Flash via OpenRouter, free) + guards + merge with rules | ✅ |
| Grounded LLM answers with automatic grounding check (withheld if ungrounded) | ✅ |
| Evaluation pipeline + report | ✅ |
| Grounded LLM answers with automatic grounding check (withheld if ungrounded) | ✅ |
| Streamlit UI (chat + product browser, LLM on/off toggle) | ✅ |

## Architecture

```
query ─► LLM parser (guarded, merged with rule parser; rules-only fallback) ─► StructuredQuery
      ─► validate against real catalogue values
      ─► HARD FILTERS (code-enforced) ─► eligible products
      ─► dense search (ChromaDB, filtered) ┐
      ─► BM25 (eligible products only)     ┴► Reciprocal Rank Fusion ─► [optional reranker]
      ─► small boosts (skin-type tags, rating) ─► variant + brand diversity
      ─► final constraint re-check ─► full parent documents
      ─► reviews searched ONLY within each product ─► balanced evidence + rating distribution
      ─► grounded LLM answer ─► automatic grounding check (withhold if it fails) ─► UI
```

## Setup (Windows / PowerShell)

```powershell
cd sephra
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .          # makes `import sephra` work from any folder
```

The processed data (`data/processed/`) and vector database
(`data/chroma_db/`) must exist. They are produced once by the pipeline
scripts in `src/`: `clean_data.py`, `create_documents.py`,
`prepare_documents.py`, `analyze_review_coverage.py`, and then
`build_vector_db.py --rebuild`. **Don't run the last one unless you mean
to**: it deletes and re-embeds about 60k chunks.

**Optional LLM:** copy `.env.example` to `.env` and add a free OpenRouter
key. Without it, Sephra runs in retrieval-only mode. The free tier has a
daily request cap. When it's reached, Sephra falls back to rule-based
parsing and shows the evidence without an AI-written answer.

## Usage

```powershell
.venv\Scripts\streamlit.exe run app.py                       # the app
.venv\Scripts\python.exe -m pytest                            # 141 tests (unit + read-only integration; LLM tests use fakes, no quota)
.venv\Scripts\python.exe -m pytest -m "not integration"       # fast unit tests only
.venv\Scripts\python.exe -m sephra.evaluation.evaluate        # retrieval benchmark (~5 min)
.venv\Scripts\python.exe -m sephra.evaluation.evaluate_llm    # live LLM evaluation (uses free quota; cached)
```

From Python:

```python
from sephra.retrieval.pipeline import SephraRetriever
retriever = SephraRetriever()                     # load once, reuse
response = retriever.search("Sunscreen under $25")
for result in response.products:
    print(result.product.name, result.product.price_usd, result.evidence.loaded_review_count)
```

## Project layout

```
src/sephra/
  config.py            paths and retrieval settings
  catalog.py           products + reviews loaded once (source of truth)
  embeddings.py        cached BGE model + query instruction
  vector_store.py      read-only ChromaDB access
  query/               schema.py (Pydantic), llm_parser.py, rule_parser.py, vocabulary.py, filters.py
  retrieval/           dense, bm25, fusion (RRF), reranker, ranking, variants, reviews, resolver, pipeline
  generation/          context (evidence), prompts, llm_client (OpenRouter), answer_generator, grounding
  evaluation/          relevance rubrics, metrics, evaluate.py
  ui/components.py     Streamlit rendering
src/*.py               original data-pipeline scripts (+ legacy retrieve.py, kept as the baseline)
app.py                 Streamlit app
tests/                 pytest suite
evaluation/            queries, original baseline results, benchmark outputs
docs/                  technical_decisions.md, evaluation_report.md
```

## Key results ([full report](docs/evaluation_report.md))

| | Original system | Sephra (hybrid) |
|---|---|---|
| Hard-constraint violations (20 queries) | 33 | **0** |
| nDCG@5 (16 labelled queries) | 0.607 | **0.960** |
| Critical review shown when one exists | 4/17 | **18/18** |
| No-result / review-question / comparison handled | 0/3 | **3/3** |
| Mean query latency (warm, CPU) | 0.12 s | 0.12 s |

The labels are transparent rule-based rubrics, not human judgements. Read
the report's limitations section before quoting these numbers.

## Known limitations

- The rule-based fallback understands a fixed vocabulary only. The LLM
  parser handles paraphrases, but the free model is a *preview*, has a
  daily cap, and adds about 4 s to parsing and about 8 s to answers.
- Only one of the Kaggle review partitions is loaded: 1,104 of 8,494
  products have reviews. "No reviews" means none **in our loaded data**.
- The dataset has no verified image URLs, so cards are text-only.
- "Gentle", "not greasy" and similar qualities have no structured signal
  and rely on semantic search.
- The first query after start-up takes about 7 s while models load.
