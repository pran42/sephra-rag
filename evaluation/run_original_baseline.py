"""Capture the ORIGINAL retrieval baseline (src/retrieve.py, unmodified).

This script imports the original retrieve.py exactly as it was before the
refactor and runs it over every evaluation query. The output is saved to
evaluation/results/baseline_original.json so later versions can be compared
against the real starting point.

Run from anywhere:
    .venv\\Scripts\\python.exe evaluation\\run_original_baseline.py
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import retrieve as original  # noqa: E402  (the untouched legacy module)

QUERIES_PATH = PROJECT_ROOT / "evaluation" / "queries.json"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "results" / "baseline_original.json"


def main() -> None:
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    records = []

    for item in queries:
        start = time.perf_counter()
        results = original.retrieve_products_with_reviews(item["query"])
        elapsed = time.perf_counter() - start

        products = []
        for result in results:
            product = result["product"]
            meta = product["metadata"]
            products.append({
                "product_id": product["product_id"],
                "brand_name": meta.get("brand_name"),
                "primary_category": meta.get("primary_category"),
                "secondary_category": meta.get("secondary_category"),
                "price_usd": meta.get("price_usd"),
                "rating": meta.get("rating"),
                "distance": product["distance"],
                "review_count": result["review_count"],
                "reviews": [
                    {
                        "review_id": review["review_id"],
                        "product_id": review["metadata"].get("product_id"),
                        "rating": review["metadata"].get("rating"),
                    }
                    for review in result["reviews"]
                ],
            })

        records.append({
            "id": item["id"],
            "type": item["type"],
            "query": item["query"],
            "latency_seconds": round(elapsed, 3),
            "products": products,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nSaved original baseline for {len(records)} queries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
