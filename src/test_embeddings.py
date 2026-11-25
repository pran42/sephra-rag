
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully!")

# Test sentences
sentences = [
    "Best moisturizer for dry skin",
    "Hydrating cream suitable for dry skin",
    "Best perfume for women"
]

# Generate embeddings
embeddings = model.encode(
    sentences,
    normalize_embeddings=True
)

print("\n========== EMBEDDING RESULTS ==========")

print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)
print("Vector dimensions:", len(embeddings[0]))

# Calculate similarity
similarity = np.dot(embeddings[0], embeddings[1])

print("\nSimilarity between first two sentences:", similarity)


# ==================================
# CHECK DOCUMENT TOKEN LENGTHS
# ==================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data/processed"

token_lengths = []

for filename in [
    "products_prepared.jsonl",
    "reviews_prepared.jsonl"
]:

    with open(DATA_DIR / filename, "r", encoding="utf-8") as file:

        for line in file:
            doc = json.loads(line)

            tokens = model.tokenizer(
                doc["text"],
                add_special_tokens=True,
                truncation=False
            )["input_ids"]

            token_lengths.append(len(tokens))


print("\n========== TOKEN LENGTH ANALYSIS ==========")

lengths = np.array(token_lengths)

print("Total documents:", len(lengths))
print("Average tokens:", round(lengths.mean(), 2))
print("95th percentile:", int(np.percentile(lengths, 95)))
print("Maximum tokens:", lengths.max())
print("Documents exceeding 512 tokens:", np.sum(lengths > 512))
