
import argparse
import json
import sys
from pathlib import Path

# ------------------------------------------------------------------
# SAFETY GUARD (added during the refactor)
# This script DELETES both collections and re-embeds ~60,000 chunks.
# It must never run by accident, so it requires an explicit flag.
# Back up data/chroma_db first if you really want to rebuild.
# ------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="Rebuild the Sephra ChromaDB from scratch.")
_parser.add_argument("--rebuild", action="store_true",
                     help="Required. Deletes and re-creates both collections.")
if not _parser.parse_args().rebuild:
    print("Refusing to run: this would DELETE and rebuild the existing vector database.\n"
          "If you really mean it, back up data/chroma_db and run:\n"
          "    python src/build_vector_db.py --rebuild")
    sys.exit(1)

import chromadb
from sentence_transformers import SentenceTransformer


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
DB_DIR = BASE_DIR / "data" / "chroma_db"

# Embedding model
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Settings
BATCH_SIZE = 64

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# Create persistent ChromaDB client
client = chromadb.PersistentClient(path=str(DB_DIR))


def create_collection(name):
    """Create a fresh collection using cosine similarity."""

    existing = [c.name for c in client.list_collections()]

    if name in existing:
        client.delete_collection(name)

    return client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )


def index_documents(collection, filename):
    """Read JSONL documents in batches and store their embeddings."""

    file_path = DATA_DIR / filename

    batch = []
    total_indexed = 0

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:
            record = json.loads(line)
            batch.append(record)

            if len(batch) == BATCH_SIZE:
                add_batch(collection, batch)
                total_indexed += len(batch)
                batch = []

                print(f"Indexed: {total_indexed}")

    # Process remaining documents
    if batch:
        add_batch(collection, batch)
        total_indexed += len(batch)

    print(f"Completed: {total_indexed} documents")


def add_batch(collection, batch):

    ids = [item["id"] for item in batch]
    texts = [item["text"] for item in batch]
    metadatas = [item["metadata"] for item in batch]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )


# Create product collection
print("\nCreating product collection...")
product_collection = create_collection("sephra_products")

index_documents(
    product_collection,
    "products_prepared.jsonl"
)

# Create review collection
print("\nCreating review collection...")
review_collection = create_collection("sephra_reviews")

index_documents(
    review_collection,
    "reviews_prepared.jsonl"
)

print("\nVector database creation completed!")

print("Product vectors:", product_collection.count())
print("Review vectors:", review_collection.count())
