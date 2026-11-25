# Sephra: Technical Decisions

Each section covers what a component does, why it was chosen, the
alternatives, the trade-offs, and how to test the choice. Measured results
are in [evaluation_report.md](evaluation_report.md). No parameter here is
claimed to be optimal. Where something was not tested, this document says
so.

---

## 1. Embedding model: BAAI/bge-small-en-v1.5

- **What it does.** It turns text (product documents, reviews, queries)
  into vectors, so that texts with similar meaning end up close together.
- **Why this model:**
  - It is small (about 33M parameters, about 130 MB), so it runs on a
    laptop CPU at about 20 ms per query.
  - It ranked well for its size on the MTEB retrieval benchmark when
    chosen.
  - It is English-only, which matches the dataset.
- **Alternatives:**
  - `bge-base` or `bge-large`: probably more accurate, but 3–10× slower.
  - `all-MiniLM-L6-v2`: faster, but weaker on retrieval.
  - OpenAI or Cohere embeddings: paid, need the network, and the project
    avoids paid APIs.
- **Trade-off.** Speed and zero cost versus some accuracy. Changing models
  means re-embedding all 60,647 chunks, so the model is fixed in
  `config.py`.
- **How to test.** Embed the products only (10.6k chunks) into a *separate*
  collection with another model, then run the same evaluation.

## 2. Why 384 dimensions

The vector length is fixed by the model's architecture: bge-small's
hidden size is 384. It isn't a setting. Smaller vectors mean less storage
(60k × 384 floats is about 90 MB) and faster similarity maths, though they
can encode less nuance than 768- or 1024-dimension models.

## 3. Why 480 tokens per chunk

- bge-small reads at most **512 tokens**; anything longer is cut off
  silently.
- 480 leaves room for the special tokens and a safety margin for
  differences between tokenizer counts.
- With this limit, 1,023 of 8,494 products needed more than one chunk (up
  to 13), and only 94 of 49,918 reviews did. Ingredient lists are the
  long part.
- **Alternatives.** Smaller chunks (128–256 tokens) are more precise but
  split product facts apart. Larger chunks aren't possible with this model.

## 4. Chunking approach (and a known weakness)

- **What it does.** `prepare_documents.py` splits text at sentence
  boundaries, packs whole sentences up to 480 tokens, and only falls back
  to splitting by token count for single sentences that are too long.
- **Why.** It keeps reviews whole (99.8% are a single chunk) and keeps
  sentences intact.
- **Known weakness (found in the audit, not fixed on purpose):**
  - Product documents put the ingredients *before* the highlights.
  - So 2,140 continuation chunks carry no product name, and 821 of them
    hold the highlight tags.
  - Normalisation also removes apostrophes ("don't" becomes "dont").
- **Why not fixed.**
  - Fixing it means re-embedding the product chunks.
  - Parent-document retrieval and BM25 over full records already make up
    for it, and the evaluation shows strong product recall (capped
    R@50 = 0.93).
- **How to test.** Rebuild the product collection with "title + highlights
  first" chunks into a new collection, then compare dense nDCG.

## 5. ChromaDB

- **Why.** It is embedded (no server), persistent, has a simple Python
  API, and supports metadata `where` filters (`$and`, `$or`, `$in`, `$lte`
  and so on).
- **Alternatives:**
  - FAISS: fast, but no metadata filtering and no persistence of its own.
  - Qdrant or Weaviate: production-grade, but they need a server.
  - pgvector: needs PostgreSQL.
- **Trade-off.** Ideal for 60k vectors on one machine. It isn't built for
  multi-user scale.
- **Safety.** `vector_store.py` only reads. `build_vector_db.py` now
  refuses to run without `--rebuild`, because it deletes both collections.

## 6. Cosine distance

- The collections use `hnsw:space = cosine`, and embeddings are
  normalised to length 1.
- For unit vectors, cosine similarity equals the dot product, so the
  ranking is the same either way.
- Cosine was chosen because BGE is trained with cosine similarity, and it
  ignores vector length, which carries no meaning here.
- Sephra reports similarity as `1 − distance`.

## 7. The BGE query instruction

- Queries are prefixed with "Represent this sentence for searching
  relevant passages: ". Documents are not.
- BGE was trained this way for short queries against longer passages.
  The instruction tells the model to encode an *information need*, not a
  passage.
- **How to test.** Run the evaluation with the instruction removed. This
  costs nothing because only the query embeddings change.

## 8. Why structured metadata filtering is necessary

- An embedding of "under $30" doesn't stop a $70 cream from being
  semantically close. The original system returned it at rank #4.
- Hard requirements must be enforced by code. There is one function,
  `product_violations`, and it is used:
  1. to build the eligible set,
  2. as a final safety check on every result,
  3. in the evaluation checks.
- **Result:** 33 constraint violations went down to 0.
- **How it reaches ChromaDB:**
  - **Native metadata filters** when Chroma stores the field: price,
    rating, primary and secondary category, brand, stock. Example:
    `{"$and": [{"price_usd": {"$lte": 30}}, {"rating": {"$gte": 4}}]}`.
  - **A `product_id $in [...]` list** when it doesn't: the tertiary
    category (Shampoo, Face Serums, Perfume), highlight tags, ingredients.
    The catalogue computes the eligible IDs first.
- **No silent relaxing.** When nothing matches, the user sees which
  constraint removed the last products, for example "brand Tatcha: 6 →
  price ≤ $10: 0".
- **Product vs. reviewer skin type.**
  - The *product's* suitability comes from listing tags such as "Best for
    Dry, Combo, Normal Skin". It is a soft ranking boost, because 26% of
    products have no tags.
  - A *reviewer's* skin type is only used to slightly prefer reviews from
    people with the same skin type.

## 9. Why dense retrieval alone is not enough

- It can't enforce numbers, as shown above.
- It blurs exact terms: brand names, product types, tags like "Oil Free".
- It can rank a slightly off-topic product above an exact match.

## 10. Why BM25 helps

- BM25 scores exact word matches, weighted by how rare a word is (IDF),
  with saturation and length normalisation.
- It catches what dense search blurs: "shampoo", "Drunk Elephant",
  "vegan", "SPF".
- **What is indexed.** Product name (counted twice, as a cheap field
  boost), brand, category path and highlight tags.
- **What isn't.** Ingredient lists. They are long and would dominate
  length normalisation, and ingredient requests are handled by a hard
  filter.
- It is built in memory in about 1 s, with no new embeddings.

## 11. Why hybrid retrieval

Dense search finds meaning ("hydrating but not greasy" lines up with "gel
cream"), and BM25 finds exact terms. Combining them gives better coverage
than either alone. Measured: hybrid's capped R@50 is 0.927 vs 0.903 for
dense, and strict P@5 is 0.86 vs 0.84. These gaps are small and within
noise on 16 queries.

## 12. Reciprocal Rank Fusion (RRF, k = 60)

- **Formula:** `score = Σ 1/(k + rank)`.
- **Why it's needed.** Dense similarities (about 0.6–0.9) and BM25 scores
  (0 to 30+) are on different scales, so adding them is meaningless. RRF
  only uses ranks, so no calibration is needed.
- **Why k = 60.** It comes from the original paper (Cormack et al., 2009)
  and was not tuned here.
- **Alternative.** A weighted sum of normalised scores. It needs a weight
  that has to be tuned, and it is sensitive to the score distribution.

## 13. Parent-document retrieval

- Search runs over *chunks*, but users need *whole* products and reviews.
- Each chunk stores a `parent_id`. Sephra searches chunks, keeps the best
  chunk per product, and returns the full original record from the
  catalogue, not a fragment.
- The catalogue also supplies fields that Chroma doesn't store: tertiary
  category, highlights as a list, size, sale price.

## 14. Product-specific review retrieval

- Reviews are **only** searched with a mandatory `product_id` filter, and
  each result is checked against the catalogue again before use.
- Searching reviews on their own and then attaching them to products
  would pair, say, a La Mer review with a Tatcha recommendation.
- Result: 0 misattributed reviews across all evaluations.
- Products with 0 loaded reviews stay eligible. The note explains that
  reviews may exist in Kaggle review partitions we didn't load, and quotes
  the product record's own Sephora review count where one exists.

## 15. Balanced review evidence

- Sephra picks the most relevant review from each sentiment bucket that
  *exists* for the product: positive (4–5★), mixed (3★) and critical
  (1–2★).
- It runs a targeted search if a bucket is missing from the top 30
  candidates, then fills the remaining slots by relevance.
- It never forces a 50/50 split.
- Excerpts always appear next to the rating distribution of *all* the
  product's loaded reviews. They are labelled "not a representative
  sample".
- Result: a critical review was shown for every product that has one,
  compared with 4 of 17 in the baseline.

## 16. Deduplication and product variants

- **Chunk deduplication.** A product with several matching chunks takes
  one slot. The best chunk wins.
- **Variant families.** For example, P411403 Confidence in a Cream ($20,
  4.62★, 0 reviews) and P427641, the Mini ($20, 3.68★, 47 reviews), have
  different data and must not be merged. They are grouped (same brand
  required) when:
  - they share identical review texts (the dataset copies reviews across
    sizes; 984 such reviews were found), or
  - they are in the same category with near-identical names (size words
    such as "mini" or "travel" are ignored).
- The top 5 gets one slot per family. Other eligible members are shown
  as "related variants" with their own ID, price, rating and reviews.
- **Brand cap.** At most 2 products per brand, unless the user asked for
  a brand. This was added after "perfume" returned 5 products from one
  brand.
- **Limitation.** Grouping is heuristic. Flankers such as "Good Girl" and
  "Very Good Girl" end up in one family, which is why the UI says
  "related" rather than "same product".

## 17. Why reranking *might* help

- A cross-encoder reads the query and the product *together*, which is
  more accurate in principle than comparing two separately computed
  vectors.
- It is slow, so it only reranks the top 25 candidates.

## 18. Reranker selection (benchmarked)

| Model | Parameters | nDCG@5 | Latency per query (CPU) |
|---|---|---|---|
| none (hybrid) | n/a | 0.960 | 0.12 s |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | ~22M | 0.943 | 1.79 s |
| BAAI/bge-reranker-base | ~278M | 0.961 | 9.72 s |

**Decision: no reranker by default.** It is configurable through
`mode="hybrid_rerank"` and `reranker="minilm" | "bge-base"`. Neither model
improved quality measurably, and both add a lot of latency. The models
were trained on web question answering (MS MARCO), not product
catalogues.

## 19. LLM provider, query understanding and answer generation

**Provider and model.** OpenRouter, using `baidu/ernie-5.5-flash-preview:free`.
- **Why:** the free tier fits the "no paid API" requirement. The model
  supports strict JSON-schema output, which suits query parsing, and has a
  1M-token context.
- **Trade-offs:**
  - It is a *preview* model and may be withdrawn, so the model ID lives in
    `.env`.
  - The free tier has a daily request cap.
  - Measured latency: about 4 s per parse and about 7.7 s per answer
    (max 14.9 s), compared with about 0.1 s for retrieval alone.
- **Provider independence.** `llm_client.py` exposes one `complete()`
  method. The client talks plain HTTPS (`httpx`), so swapping providers
  means writing one class.

**The query parser (`query/llm_parser.py`): the LLM proposes, code
decides.**
1. The model fills a strict JSON schema. The prompt lists the real
   category paths and brands.
2. Pydantic validates the reply.
3. Deterministic guards remove anything the request doesn't support.
   Live testing found real cases:
   - prices or ratings that aren't stated in the text;
   - brands that aren't mentioned;
   - attributes, ingredients or skin types with no supporting word. ERNIE
     added "Vegan" and "Cruelty-Free" to a dry-skin moisturizer query, and
     4 invented attributes to a lipstick query. It also added "Oil Free"
     for "oily skin", which exposed a bug in the guard itself.
4. Categories that don't exist, or match no product, trigger one
   corrective retry. Truncated JSON triggers one retry.
5. **Merging with the rule parser.** The LLM sometimes *dropped*
   constraints stated as plain keywords, for example "cleanser" and
   "skincare". So the rule parser's explicit keyword matches are merged
   in: rule categories take priority, missing prices and ratings are
   filled in, and attributes and skin types are combined.
6. Any failure (no key, rate limit, invalid output) falls back to the rule
   parser, and the reason is recorded in `notes`.
7. A circuit breaker stops retries for 10 minutes after a daily-limit
   error.

**Answer generation (`generation/answer_generator.py`).**
- The model sees only the evidence text assembled by `context.py`, under
  the rules in `prompts.py` (no invented facts, opinions attributed to
  reviewers, missing reviews stated, no medical claims).
- Each answer passes through `grounding.py`, which checks that every
  price, decimal rating and product ID appears in the evidence.
- If a check fails, the model gets one retry with the problems listed. If
  it still fails, the answer is **withheld** and the evidence is shown
  instead.
- No LLM call is made for no-result responses, to avoid risk and save
  quota.

**Results.** See [evaluation_report.md §6](evaluation_report.md).
- LLM-only parsing scored slightly below the rule parser on the keyword
  benchmark (nDCG 0.949 vs 0.960).
- LLM merged with rules matched it (0.960) and blocked 8 invented
  constraints.
- 32 of 32 generated answers passed grounding on the first attempt.

## 20. Top-k and candidate sizes

| Setting | Value | Reasoning |
|---|---|---|
| Products shown | 5 | A readable answer; also the original baseline value |
| Dense chunks fetched | 100 | Products have up to 13 chunks; 100 yields about 60–90 unique products |
| BM25 candidates | 50 | Enough for fusion; recall is measured at 50 |
| Rerank window | 25 | Bounds cross-encoder cost |
| Review chunks per product | 30 | Room to find each sentiment; plus targeted searches |
| Reviews shown per product | 4 (6 for review questions) | One per sentiment, plus the most relevant |
| Skin-tag boost / rating boost | ≤ 0.10 / ≤ 0.05 | Small on purpose: relevance (0–1) dominates. Not tuned. |

None of these were tuned. They are reasonable starting points, and the
evaluation script can test changes to them.

## 21. How retrieval quality is evaluated

See [evaluation_report.md](evaluation_report.md). In short:
- 20 queries.
- Rule-based relevance labels across the whole catalogue (the
  limitations are documented there).
- Precision@5, strict P@5, MRR, nDCG@5 and capped Recall@50.
- Label-free checks: constraint violations, duplicate families,
  misattributed reviews, zero-review notes, critical-evidence coverage,
  special-case outcomes.
- Latency.
- A comparison against the **original** system's results, captured
  before any changes.
- A parity test, which proves the refactor initially reproduced the
  original output exactly.

## 22. Other alternatives considered

| Alternative | Why it wasn't used |
|---|---|
| Asking the LLM to "respect the budget" | Not guaranteed; constraints must be code |
| Searching reviews on their own and joining them to products | Risks misattribution, which the brief rules out |
| Merging variants into one product | Loses the differences in price, rating and reviews |
| Ranking by review count | Says nothing about suitability, unfairly penalises zero-review products, and reflects our partial dataset |
| Rebuilding embeddings with better chunking | Expensive; current recall is already high; left as a measured future experiment |
| LangChain or LlamaIndex | More dependencies and abstraction than this pipeline needs; harder to explain line by line |
