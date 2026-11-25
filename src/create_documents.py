
import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data/processed"

products = pd.read_csv(DATA_DIR / "products_clean.csv")
reviews = pd.read_csv(DATA_DIR / "reviews_clean.csv")


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def make_record(doc_id, text, row, metadata_columns):
    metadata = {}

    for col in metadata_columns:
        value = row[col]

        if pd.isna(value) or str(value).strip() == "":
            continue

        if hasattr(value, "item"):
            value = value.item()

        metadata[col] = value

    return {
        "id": str(doc_id),
        "text": text,
        "metadata": metadata
    }


# ==================================
# 1. CREATE PRODUCT DOCUMENTS
# ==================================

product_documents = []

for _, row in products.iterrows():

    fields = {
        "Product": row["product_name"],
        "Brand": row["brand_name"],
        "Category": row["primary_category"],
        "Subcategory": row["secondary_category"],
        "Price (USD)": row["price_usd"],
        "Rating": row["rating"],
        "Ingredients": row["ingredients"],
        "Highlights": row["highlights"]
    }

    text = "\n".join(
        f"{key}: {clean(value)}"
        for key, value in fields.items()
        if clean(value) != ""
    )

    metadata_columns = [
        "product_id",
        "brand_name",
        "primary_category",
        "secondary_category",
        "price_usd",
        "rating",
        "out_of_stock"
    ]

    record = make_record(
        f"product_{row['product_id']}",
        text,
        row,
        metadata_columns
    )

    product_documents.append(record)


# ==================================
# 2. CREATE REVIEW DOCUMENTS
# ==================================

reviews = reviews.merge(
    products[["product_id", "primary_category"]],
    on="product_id",
    how="left"
)

review_documents = []

for index, row in reviews.iterrows():

    fields = {
        "Product": row["product_name"],
        "Brand": row["brand_name"],
        "Category": row["primary_category"],
        "Rating": row["rating"],
        "Skin Type": row["skin_type"],
        "Review": row["review_content"]
    }

    text = "\n".join(
        f"{key}: {clean(value)}"
        for key, value in fields.items()
        if clean(value) != ""
    )

    metadata_columns = [
        "product_id",
        "brand_name",
        "primary_category",
        "rating",
        "skin_type",
        "is_recommended"
    ]

    record = make_record(
        f"review_{index}",
        text,
        row,
        metadata_columns
    )

    review_documents.append(record)


# ==================================
# 3. SAVE DOCUMENTS AS JSONL
# ==================================

def save_jsonl(documents, filename):

    path = DATA_DIR / filename

    with open(path, "w", encoding="utf-8") as file:
        for document in documents:
            file.write(
                json.dumps(document, ensure_ascii=False) + "\n"
            )

    print(f"Saved: {path}")


save_jsonl(product_documents, "product_documents.jsonl")
save_jsonl(review_documents, "review_documents.jsonl")


# ==================================
# 4. DISPLAY SUMMARY
# ==================================

print("\n========== DOCUMENT SUMMARY ==========")

print("Product documents:", len(product_documents))
print("Review documents:", len(review_documents))

print("\nProduct document example:")
print(json.dumps(product_documents[0], indent=2, ensure_ascii=False))

print("\nReview document example:")
print(json.dumps(review_documents[0], indent=2, ensure_ascii=False))

print("\n========== TEXT LENGTH ==========")

for name, documents in [
    ("Products", product_documents),
    ("Reviews", review_documents)
]:
    lengths = pd.Series([len(d["text"]) for d in documents])

    print(f"\n{name}:")
    print("Average characters:", round(lengths.mean(), 2))
    print("95th percentile:", int(lengths.quantile(0.95)))
    print("Maximum characters:", lengths.max())
