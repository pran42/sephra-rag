# Sephra: Retrieval Evaluation Report

All numbers below come from `python -m sephra.evaluation.evaluate`. The
raw results are in `evaluation/results/evaluation_results.json`, and an
auto-generated table is in `evaluation/results/evaluation_summary.md`.
Measured on Windows 11 (CPU only), Python 3.14.

## 1. What was evaluated

**Queries:** 20 queries in `evaluation/queries.json`. They cover the 14
required query types plus 6 extra retrieval queries.

| ID | Type | Query |
|---|---|---|
| Q01 | dry skin + budget | Moisturizer for dry skin under $30 that is hydrating and not greasy |
| Q02 | fragrance-free | Fragrance-free skincare products |
| Q03 | oily, lightweight | Lightweight moisturizer for oily skin |
| Q04 | sunscreen + budget | Sunscreen under $25 |
| Q05 | highly rated | Highly rated shampoo for damaged hair |
| Q06 | perfume | Perfume for everyday use |
| Q07 | brand | Show me skincare products from Drunk Elephant |
| Q08 | zero-review products | Hydrating oil-free gel moisturizer under $30 |
| Q09 | positive + negative reviews | What do customers say about the Tata Harper Repairative Anti-Aging Moisturizer? |
| Q10 | no match | Tatcha moisturizer under $10 |
| Q11 | strict max price | Face serum under $15 |
| Q12 | no budget | Vitamin C serum for brightening |
| Q13 | ambiguous | I need something nice |
| Q14 | comparison | Compare La Mer The Moisturizing Cool Gel Cream and Tata Harper Repairative Anti-Aging Moisturizer |
| Q15–Q20 | extra | hyaluronic serum, vegan mascara, hair oil for frizz, body lotion for very dry skin, long-lasting matte lipstick, gentle cleanser for sensitive skin |

**Systems compared:**

| System | Description |
|---|---|
| `baseline_original` | The original `src/retrieve.py`: dense search only, no filters. Its results were captured before any refactoring. |
| `dense` | Hard filters + dense retrieval |
| `hybrid` | Hard filters + dense + BM25, fused with Reciprocal Rank Fusion **(default)** |
| `hybrid_rerank_minilm` | hybrid + `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `hybrid_rerank_bge-base` | hybrid + `BAAI/bge-reranker-base` |

## 2. How relevance labels were created

The labels in `src/sephra/evaluation/relevance.py` are **transparent,
rule-based rubrics**, one hand-written function per query. Each function
grades **every product in the catalogue** using only verifiable dataset
fields: category, price, rating, highlight tags, the product name and the
ingredient list.

- Grade 0 means not relevant, 1 means relevant and 2 means highly relevant.
- Example (Q01): the product must be a face moisturizer at ≤ $30 to get
  grade 1. It gets grade 2 if it also has a dry-skin signal and a
  hydration signal.
- Retrieval systems and LLMs played no part in creating the labels.
- Because the whole catalogue is labelled, recall is exact and no pooling
  is needed.
- **Limitations.** The rubrics measure whether the *listing claims* a
  property, not whether the product really has it. "Not greasy" and
  "gentle" can't be checked against structured data at all. The rubrics
  read highlight tags and names, which BM25 also indexes, so they may
  favour keyword retrieval slightly. They were written by the developer
  with AI assistance and **should be reviewed by a human**.
- Q09, Q10 and Q14 are scored by **expected outcome** instead: the correct
  named product(s), or an honest "no results".
- Q13 is deliberately unlabelled because it is ambiguous.

## 3. Results

**Ranking quality** is averaged over the 16 labelled queries (K = 5).
Strict P@5 counts only grade-2 products, averaged over the 10 queries whose
rubric has a grade 2.

| System | P@5 | strict P@5 | MRR | nDCG@5 | capped R@50 | Mean latency (warm) |
|---|---|---|---|---|---|---|
| baseline_original | 0.650 | 0.620 | 0.771 | 0.607 | n/a | 0.12 s |
| dense | 1.000 | 0.840 | 1.000 | 0.953 | 0.903 | 0.25 s* |
| **hybrid (default)** | 1.000 | **0.860** | 1.000 | 0.960 | 0.927 | **0.12 s** |
| hybrid + MiniLM rerank | 1.000 | 0.820 | 1.000 | 0.943 | 0.934 | 1.79 s |
| hybrid + bge-reranker-base | 1.000 | 0.840 | 1.000 | 0.961 | 0.934 | 9.72 s |

\*`dense` ran first, so its latency probably includes cold caches.
Latency ordering between dense and hybrid should not be read as
meaningful.

**Correctness checks.** These are automatic and need no labels. Counts are
summed over all 20 queries.

| Check | baseline_original | All new systems |
|---|---|---|
| Hard-constraint violations in returned products | **33** | **0** |
| Duplicate variant families in the same top 5 | 4 | 0 |
| Reviews attached to the wrong product | 0 | 0 |
| Zero-review products shown without an explanation | n/a | 0 |
| Critical review shown when the product has critical reviews | 4 of 17 | all (17–20 of 17–20) |
| Special cases (no-result, named review question, comparison) | 0 of 3 | 3 of 3 |

## 4. Interpretation (and what is NOT claimed)

1. **The big win is constraint satisfaction, not "smarter search".** The
   original system returned products that broke the user's explicit
   requirements 33 times. One example is the $70 Tatcha cream for "under
   $30". Hard filters reduce that to 0 by construction.
2. **P@5 saturates at 1.0** for every filtered system, because the rubric's
   grade-1 conditions overlap with the hard filters. The more informative
   comparisons are strict P@5 and nDCG@5.
3. **Dense vs. hybrid:** hybrid scores slightly higher (strict P@5 0.86 vs
   0.84, nDCG 0.960 vs 0.953, capped R@50 0.927 vs 0.903). With 16 queries
   these gaps are **within noise**. The claim is only "hybrid is at least
   as good, and fast".
4. **Rerankers did not earn their cost.**
   - MiniLM was slightly worse overall (nDCG 0.943). Its largest drop was on
     Q19, the matte lipstick query (0.91 → 0.70), and it is about 15×
     slower.
   - bge-reranker-base matched hybrid (0.961 vs 0.960) at about 80× the
     latency on CPU.
   - **Decision: the reranker stays optional and off by default.**
   - Likely reason: the MS MARCO-trained cross-encoders were trained on web
     question answering, not short product listings, and the candidates are
     already filtered to the right category.
5. **Weakest query: Q20 ("gentle cleanser for sensitive skin")**, with
   nDCG 0.45–0.56. "Gentle" and "sensitive" have no structured tag. The
   dataset has no "Best for Sensitive Skin" tag, so the fragrance-free tag
   is only a weak stand-in.
6. **Small sample.** 20 queries is enough to catch correctness bugs, but
   not enough to establish statistically reliable differences between
   rankers.

## 5. Reproducing

```powershell
.venv\Scripts\python.exe -m sephra.evaluation.evaluate                 # all systems (~5 min)
.venv\Scripts\python.exe -m sephra.evaluation.evaluate --systems dense hybrid
.venv\Scripts\python.exe evaluation\run_original_baseline.py           # re-capture original baseline
```

## 6. LLM layer (live, ERNIE 5.5 Flash via OpenRouter free tier)

`python -m sephra.evaluation.evaluate_llm` runs the same 20 queries
through hybrid retrieval twice: once with the rule parser and once with the
LLM parser. It also generates an answer for every query with results.
Results are in `evaluation/results/llm_evaluation.json`, and replies are
cached in `llm_cache.json`.

**Query understanding** (16 labelled queries):

| Parser | P@5 | nDCG@5 | Constraint drops vs. rule reading | Invented constraints blocked by guards | Fallbacks |
|---|---|---|---|---|---|
| Rule-based | 1.000 | 0.960 | n/a | n/a | n/a |
| LLM only (first run) | 0.988 | 0.949 | 2 (Q08, Q20) | 7 | 0 |
| **LLM + guards + merge with rules (current)** | 1.000 | 0.960 | 0 | 8 | 0 |

**What the first run showed:**
- **Invented constraints.** The LLM added hard filters nobody asked for,
  such as "Vegan" and "Cruelty-Free" on a dry-skin moisturizer query, and 4
  attributes on a lipstick query. The guards removed them. One slipped
  through: "Oil Free" for "oily skin", caused by a guard regex bug, now
  fixed and tested.
- **Dropped constraints.** It sometimes left out constraints that were
  stated plainly: Q02 lost "skincare", Q08 lost "moisturizer" and
  "oil-free", Q20 lost "cleanser". This is why the parse is now merged with
  the rule parser's explicit keyword matches.

**Honest interpretation:**
- On this benchmark the merged parser only *matches* the rule parser.
- The benchmark queries use the keyword phrasing the rule vocabulary was
  built from, so they can't show where an LLM helps: paraphrases. In
  manual spot checks, it correctly read "no perfume please" as Fragrance
  Free and "won't break the bank" as *no* stated budget. The rule parser
  can't do either.
- A paraphrase test set is the next evaluation step. It was postponed
  because the free daily quota ran out.

**Answer generation:**

| Run | Generated | Passed grounding on first attempt | Withheld | Rate-limited | Skipped (no results) |
|---|---|---|---|---|---|
| Run 1 | 19 | 19 | 0 | 0 | 1 |
| Run 2 (after the merge changed some evidence) | 13 | 13 | 0 | 6 (daily free cap) | 1 |

- Every generated answer (32 of 32) passed the automatic grounding check:
  no unsupported prices, decimal ratings or product IDs. The check doesn't
  verify every sentence, for example paraphrased review opinions.
- Manually reviewed answers stated "0 customer reviews in our loaded
  dataset" correctly, gave the full rating distribution and included
  critical feedback.

**Latency** (real API time, excluding the evaluation's pacing pauses):
- parsing: mean 4.0 s
- answers: mean 7.7 s, max 14.9 s
- compare: about 0.1 s for retrieval alone

**Free-tier limitation.** OpenRouter caps free-model requests per day on
accounts without purchased credits. When the cap is hit, the app falls back
to rule-based parsing, shows the evidence without an AI-written answer, and
explains why. A circuit breaker stops further calls for 10 minutes.

## 6b. RAGAS evaluation

`python -m sephra.evaluation.evaluate_ragas`, with `--llm` for the
LLM-judged metrics. Results are in `evaluation/results/ragas_results.json`.

**Retrieval (RAGAS ID-based metrics, no LLM needed).** Reference IDs come
from the rubric labels in §2. RAGAS was run with the system's own retrieved
IDs.

| System | Context precision | Context precision (highly relevant only) | Named-product precision / recall | Review attribution precision |
|---|---|---|---|---|
| baseline_original | 0.65 | 0.62 | 0.30 / 1.00 | n/a |
| dense | 1.00 | 0.84 | 1.00 / 1.00 | n/a |
| **hybrid** | **1.00** | **0.86** | **1.00 / 1.00** | **1.00** (86 reviews) |

These numbers match the project's own metric code, which independently
confirms the evaluation. Recall isn't reported for broad queries: with
hundreds of relevant products, recall@5 is meaningless.

**Generation (RAGAS LLM-judged metrics, judge: ERNIE 5.5 Flash).**

| Metric | Score | Answers scored |
|---|---|---|
| Faithfulness (share of answer claims supported by the retrieved context) | **0.99** | 8 of 13 |
| Answer relevancy | **0.88** | 7 of 13 |

**Caveats:**
- The free daily cap stopped the run after 8 and 7 answers.
- The judge is the same model that wrote the answers, which risks
  self-evaluation bias. An independent judge would be stronger.
- With so few samples, treat these as indicative. They agree with the
  project's own grounding check, which 32 of 32 answers passed.

## 7. Next steps for evaluation

- Have a person review the rubric grades, starting with Q03, Q19 and Q20.
- Add 20–30 more queries, especially paraphrases the rule parser doesn't
  cover. This matters before the LLM parser is compared with the rule
  parser.
- Once OpenRouter answer generation exists, run
  `sephra.generation.grounding.check_answer_grounding` on every generated
  answer and report the grounding failure rate.
