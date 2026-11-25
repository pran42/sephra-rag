"""Central configuration: file paths, model names and retrieval settings.

All paths are derived from this file's location, so every script works no
matter which folder PowerShell is opened in.
"""

from dataclasses import dataclass
from pathlib import Path

# src/sephra/config.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma_db"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

PRODUCTS_CSV = PROCESSED_DIR / "products_clean.csv"
PRODUCT_DOCUMENTS_JSONL = PROCESSED_DIR / "product_documents.jsonl"
REVIEW_DOCUMENTS_JSONL = PROCESSED_DIR / "review_documents.jsonl"
REVIEW_COUNTS_CSV = PROCESSED_DIR / "product_review_counts.csv"

# Existing ChromaDB collections (built by src/build_vector_db.py).
PRODUCT_COLLECTION = "sephra_products"
REVIEW_COLLECTION = "sephra_reviews"

# Embedding model used to build the collections. Changing this would require
# re-embedding everything, so it is deliberately a constant.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE models were trained with this prefix on *queries* (not on documents).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class RetrievalSettings:
    """Tunable retrieval parameters, kept in one place.

    These are starting values, not proven optima. The evaluation module
    compares configurations (see docs/evaluation_report.md).
    """

    # How many products to return to the user.
    top_k: int = 5

    # Dense search: number of product CHUNKS fetched from Chroma before
    # de-duplicating to products. Products can have up to 13 chunks, so we
    # over-fetch to still end up with enough unique products.
    dense_chunk_candidates: int = 100

    # BM25: number of products kept from the keyword ranking.
    bm25_candidates: int = 50

    # Reciprocal Rank Fusion constant (60 is the value from the original
    # RRF paper by Cormack et al., 2009).
    rrf_k: int = 60

    # How many fused candidates are passed to the (optional) reranker.
    rerank_candidates: int = 25

    # Reviews: chunks fetched per product, then how many are shown.
    review_chunk_candidates: int = 30
    reviews_per_product: int = 4

    # Diversity: at most this many products per brand in the final list
    # (ignored when the user asked for a specific brand).
    max_per_brand: int = 2

    # Soft ranking boosts (small on purpose: relevance dominates).
    skin_type_tag_boost: float = 0.10
    rating_boost_max: float = 0.05


DEFAULT_SETTINGS = RetrievalSettings()
