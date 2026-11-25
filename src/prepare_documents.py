
import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data/processed"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Leave room for special tokens
MAX_TOKENS = 480

print("Loading tokenizer...")
model = SentenceTransformer(MODEL_NAME)
tokenizer = model.tokenizer


# ==========================================
# LOAD AND SAVE JSONL
# ==========================================

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


# ==========================================
# TOKEN COUNTING
# ==========================================

def count_tokens(text):
    return len(
        tokenizer(
            text,
            add_special_tokens=True,
            truncation=False
        )["input_ids"]
    )


# ==========================================
# TEXT NORMALIZATION
# ==========================================

def normalize_text(text):
    # Remove Python list brackets and quotes
    text = re.sub(r"[\[\]']", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# SPLIT TEXT AT SENTENCE BOUNDARIES
# ==========================================

def split_long_text(text, max_tokens=MAX_TOKENS):

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = (
            current_chunk + " " + sentence
        ).strip()

        if count_tokens(candidate) <= max_tokens:
            current_chunk = candidate

        else:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = sentence

            # Handle exceptionally long single sentences
            if count_tokens(current_chunk) > max_tokens:

                token_ids = tokenizer(
                    current_chunk,
                    add_special_tokens=False,
                    truncation=False
                )["input_ids"]

                for start in range(0, len(token_ids), max_tokens - 20):

                    part_ids = token_ids[
                        start:start + max_tokens - 20
                    ]

                    part = tokenizer.decode(
                        part_ids,
                        skip_special_tokens=True
                    )

                    if part.strip():
                        chunks.append(part.strip())

                current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ==========================================
# PRODUCT DOCUMENT PROCESSING
# ==========================================

def process_products(documents):

    prepared = []

    for doc in documents:

        text = normalize_text(doc["text"])

        # Product documents are kept intact if possible
        chunks = split_long_text(text)

        for index, chunk in enumerate(chunks):

            prepared.append({
                "id": f"{doc['id']}_chunk_{index}",
                "text": chunk,
                "metadata": {
                    **doc["metadata"],
                    "parent_id": doc["id"],
                    "chunk_index": index,
                    "total_chunks": len(chunks)
                }
            })

    return prepared


# ==========================================
# REVIEW DOCUMENT PROCESSING
# ==========================================

def process_reviews(documents):

    prepared = []

    for doc in documents:

        text = normalize_text(doc["text"])

        # Most reviews remain intact
        chunks = (
            split_long_text(text)
            if count_tokens(text) > MAX_TOKENS
            else [text]
        )

        for index, chunk in enumerate(chunks):

            prepared.append({
                "id": f"{doc['id']}_chunk_{index}",
                "text": chunk,
                "metadata": {
                    **doc["metadata"],
                    "parent_id": doc["id"],
                    "chunk_index": index,
                    "total_chunks": len(chunks)
                }
            })

    return prepared


# ==========================================
# MAIN EXECUTION
# ==========================================

print("\nLoading original documents...")

products = load_jsonl("product_documents.jsonl")
reviews = load_jsonl("review_documents.jsonl")

print("Processing products...")
prepared_products = process_products(products)

print("Processing reviews...")
prepared_reviews = process_reviews(reviews)

# Save prepared documents
save_jsonl(prepared_products, "products_prepared.jsonl")
save_jsonl(prepared_reviews, "reviews_prepared.jsonl")


# ==========================================
# VALIDATION
# ==========================================

all_documents = prepared_products + prepared_reviews

token_lengths = [
    count_tokens(doc["text"])
    for doc in all_documents
]

print("\n========== FINAL SUMMARY ==========")

print("Original products:", len(products))
print("Prepared product chunks:", len(prepared_products))

print("Original reviews:", len(reviews))
print("Prepared review chunks:", len(prepared_reviews))

print("Total chunks:", len(all_documents))

print("Maximum token count:", max(token_lengths))
print("Average token count:", round(sum(token_lengths) / len(token_lengths), 2))

print(
    "Documents exceeding 512 tokens:",
    sum(length > 512 for length in token_lengths)
)

print("\nPreparation completed!")
