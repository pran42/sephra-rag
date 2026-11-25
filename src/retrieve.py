
import json
import csv
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
DB_DIR = BASE_DIR / "data" / "chroma_db"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Retrieval settings
PRODUCT_TOP_K = 5
PRODUCT_CANDIDATE_MULTIPLIER = 20
REVIEWS_PER_PRODUCT = 3


# =====================================================
# 2. LOAD EMBEDDING MODEL
# =====================================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)


# =====================================================
# 3. CONNECT TO CHROMADB
# =====================================================

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

products_collection = client.get_collection(
    "sephra_products"
)

reviews_collection = client.get_collection(
    "sephra_reviews"
)

print("Product vectors:", products_collection.count())
print("Review vectors:", reviews_collection.count())


# =====================================================
# 4. LOAD ORIGINAL PARENT DOCUMENTS
# =====================================================

def load_parent_documents(filename):

    file_path = DATA_DIR / filename

    documents = {}

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            record = json.loads(line)

            documents[record["id"]] = record

    return documents


print("Loading original documents...")

product_documents = load_parent_documents(
    "product_documents.jsonl"
)

review_documents = load_parent_documents(
    "review_documents.jsonl"
)

print("Original products:", len(product_documents))
print("Original reviews:", len(review_documents))


# =====================================================
# 5. LOAD REVIEW COUNTS
# =====================================================

def load_review_counts():

    file_path = DATA_DIR / "product_review_counts.csv"

    counts = {}

    with open(file_path, "r", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            product_id = row["product_id"]

            counts[product_id] = int(
                row["review_count"]
            )

    return counts


review_counts = load_review_counts()

print("Review counts loaded:", len(review_counts))


# =====================================================
# 6. CREATE QUERY EMBEDDING
# =====================================================

def create_query_embedding(query):

    query_text = (
        "Represent this sentence for searching relevant passages: "
        + query
    )

    embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embedding.tolist()


# =====================================================
# 7. RETRIEVE UNIQUE PRODUCTS
# =====================================================

def retrieve_products(query, top_k=PRODUCT_TOP_K):

    print("\nSearching products...")

    total_chunks = products_collection.count()

    if total_chunks == 0:
        return []

    query_embedding = create_query_embedding(query)

    candidate_k = min(
        top_k * PRODUCT_CANDIDATE_MULTIPLIER,
        total_chunks
    )

    results = products_collection.query(
        query_embeddings=query_embedding,
        n_results=candidate_k
    )

    unique_products = {}
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for metadata, distance in zip(metadatas, distances):

        parent_id = metadata.get("parent_id")

        product_id = metadata.get("product_id")

        if not parent_id or not product_id:
            continue

        # Skip duplicate chunks belonging to the same product
        if product_id in unique_products:
            continue

        # Retrieve complete original product document
        parent_document = product_documents.get(parent_id)

        if parent_document is None:
            print("Missing parent document:", parent_id)
            continue

        unique_products[product_id] = {
            "product_id": product_id,
            "parent_id": parent_id,
            "text": parent_document["text"],
            "metadata": parent_document["metadata"],
            "distance": distance
        }

        if len(unique_products) == top_k:
            break

    return list(unique_products.values())


# =====================================================
# 8. RETRIEVE REVIEWS FOR ONE PRODUCT
# =====================================================

def retrieve_product_reviews(
    product_id,
    query,
    top_k=REVIEWS_PER_PRODUCT
):

    # Check review availability in our loaded dataset
    available_reviews = review_counts.get(product_id, 0)

    if available_reviews == 0:
        return {
            "review_count": 0,
            "reviews": []
        }

    query_embedding = create_query_embedding(query)

    results = reviews_collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={
            "product_id": product_id
        }
    )

    retrieved_reviews = []
    seen_review_ids = set()

    for metadata, distance in zip(
        results["metadatas"][0],
        results["distances"][0]
    ):

        parent_id = metadata.get("parent_id")

        if not parent_id or parent_id in seen_review_ids:
            continue

        seen_review_ids.add(parent_id)

        # Retrieve complete original review
        parent_document = review_documents.get(parent_id)

        if parent_document is None:
            continue

        retrieved_reviews.append({
            "review_id": parent_id,
            "text": parent_document["text"],
            "metadata": parent_document["metadata"],
            "distance": distance
        })

    return {
        "review_count": available_reviews,
        "reviews": retrieved_reviews
    }


# =====================================================
# 9. COMBINE PRODUCTS WITH THEIR OWN REVIEWS
# =====================================================

def retrieve_products_with_reviews(
    query,
    top_k=PRODUCT_TOP_K
):

    products = retrieve_products(
        query,
        top_k
    )

    final_results = []

    for product in products:

        product_id = product["product_id"]

        review_data = retrieve_product_reviews(
            product_id,
            query
        )

        final_results.append({

            "product": product,

            "review_count": review_data["review_count"],

            "reviews": review_data["reviews"]

        })

    return final_results


# =====================================================
# 10. DISPLAY RESULTS
# =====================================================

def display_results(results):

    print("\n========== SEPHRA RECOMMENDATIONS ==========")

    for index, item in enumerate(results, start=1):

        product = item["product"]

        print(f"\n{'=' * 60}")
        print(f"PRODUCT {index}")
        print(f"{'=' * 60}")

        print("Product ID:", product["product_id"])

        print("Product information:")
        print(product["text"])

        print(
            "\nAvailable reviews in loaded dataset:",
            item["review_count"]
        )

        if not item["reviews"]:

            print("No relevant customer reviews retrieved.")

            if item["review_count"] == 0:
                print("No reviews available in our loaded dataset.")

        else:

            print("\nRelevant customer reviews:")

            for review_index, review in enumerate(
                item["reviews"],
                start=1
            ):

                print(f"\nReview {review_index}")
                print("Review:", review["text"])
                print("Metadata:", review["metadata"])
                print("Distance:", review["distance"])


# =====================================================
# 11. TEST RETRIEVAL
# =====================================================

if __name__ == "__main__":

    user_query = (
        "Moisturizer for dry skin under $30 "
        "that is hydrating and not greasy"
    )

    results = retrieve_products_with_reviews(
        user_query
    )

    display_results(results)
