
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data/processed"


def load_jsonl(filename):
    documents = []

    with open(DATA_DIR / filename, "r", encoding="utf-8") as file:
        for line in file:
            documents.append(json.loads(line))

    return documents


def save_jsonl(documents, filename):
    path = DATA_DIR / filename

    with open(path, "w", encoding="utf-8") as file:
        for doc in documents:
            file.write(json.dumps(doc, ensure_ascii=False) + "\n")


def normalize_text(text):
    # Convert Python-list-like text into readable comma-separated text
    text = re.sub(r"[\[\]']", "", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_text(text, chunk_size=800, overlap=100):
    """Split long text into overlapping character chunks."""

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer ending at a sentence or word boundary
        if end < len(text):
            boundary = text.rfind(". ", start, end)

            if boundary > start + chunk_size // 2:
                end = boundary + 1
            else:
                boundary = text.rfind(" ", start, end)
                if boundary > start + chunk_size // 2:
                    end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


# ==================================
# 1. PRODUCT DOCUMENTS
# ==================================

products = load_jsonl("product_documents.jsonl")
prepared_products = []

for doc in products:

    text = normalize_text(doc["text"])

    chunks = split_text(text) if len(text) > 2000 else [text]

    for index, chunk in enumerate(chunks):

        prepared_products.append({
            "id": f"{doc['id']}_chunk_{index}",
            "text": chunk,
            "metadata": {
                **doc["metadata"],
                "parent_id": doc["id"],
                "chunk_index": index,
                "total_chunks": len(chunks)
            }
        })


# ==================================
# 2. REVIEW DOCUMENTS
# ==================================

reviews = load_jsonl("review_documents.jsonl")
prepared_reviews = []

for doc in reviews:

    text = normalize_text(doc["text"])

    chunks = split_text(text) if len(text) > 2000 else [text]

    for index, chunk in enumerate(chunks):

        prepared_reviews.append({
            "id": f"{doc['id']}_chunk_{index}",
            "text": chunk,
            "metadata": {
                **doc["metadata"],
                "parent_id": doc["id"],
                "chunk_index": index,
                "total_chunks": len(chunks)
            }
        })


# ==================================
# 3. SAVE PREPARED DOCUMENTS
# ==================================

save_jsonl(prepared_products, "products_prepared.jsonl")
save_jsonl(prepared_reviews, "reviews_prepared.jsonl")


print("\n========== PREPARATION SUMMARY ==========")

print("Original products:", len(products))
print("Prepared product chunks:", len(prepared_products))

print("Original reviews:", len(reviews))
print("Prepared review chunks:", len(prepared_reviews))

print("\nExample prepared product:")
print(prepared_products[0]["text"][:500])

print("\nExample prepared review:")
print(prepared_reviews[0]["text"][:500])
