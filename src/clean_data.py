
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input files
PRODUCT_PATH = BASE_DIR / "data/raw/product_info.csv"
REVIEW_PATH = BASE_DIR / "data/raw/reviews_1250-end.csv"

# Output directory
OUTPUT_DIR = BASE_DIR / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(series):
    """Remove extra spaces and handle missing text."""
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# ==========================================
# 1. CLEAN PRODUCTS
# ==========================================

products = pd.read_csv(PRODUCT_PATH, low_memory=False)

# Keep useful product information
product_columns = [
    "product_id",
    "product_name",
    "brand_name",
    "price_usd",
    "sale_price_usd",
    "rating",
    "reviews",
    "loves_count",
    "ingredients",
    "highlights",
    "primary_category",
    "secondary_category",
    "tertiary_category",
    "size",
    "variation_type",
    "variation_value",
    "out_of_stock",
    "online_only",
    "sephora_exclusive"
]

products = products[product_columns].copy()

# Clean text columns
product_text_columns = [
    "product_id", "product_name", "brand_name",
    "ingredients", "highlights", "primary_category",
    "secondary_category", "tertiary_category",
    "size", "variation_type", "variation_value"
]

for col in product_text_columns:
    products[col] = clean_text(products[col])

# Convert numeric columns
numeric_columns = [
    "price_usd", "sale_price_usd",
    "rating", "reviews", "loves_count"
]

for col in numeric_columns:
    products[col] = pd.to_numeric(products[col], errors="coerce")

# Remove records without essential identifiers
products = products[
    (products["product_id"] != "") &
    (products["product_name"] != "")
]

# Remove duplicate product IDs
products = products.drop_duplicates(subset=["product_id"])

# Save cleaned products
products.to_csv(
    OUTPUT_DIR / "products_clean.csv",
    index=False
)


# ==========================================
# 2. CLEAN REVIEWS
# ==========================================

reviews = pd.read_csv(REVIEW_PATH, low_memory=False)

# Keep useful review information
review_columns = [
    "product_id",
    "product_name",
    "brand_name",
    "rating",
    "is_recommended",
    "helpfulness",
    "total_feedback_count",
    "submission_time",
    "review_title",
    "review_text",
    "skin_type",
    "price_usd"
]

reviews = reviews[review_columns].copy()

# Clean text columns
review_text_columns = [
    "product_id", "product_name", "brand_name",
    "review_title", "review_text", "skin_type"
]

for col in review_text_columns:
    reviews[col] = clean_text(reviews[col])

# Combine title and review body
title = reviews["review_title"]
body = reviews["review_text"]

reviews["review_content"] = (
    title.where(title == "", title + ". ") + body
).str.strip()

# Keep reviews that contain some actual text
reviews = reviews[reviews["review_content"] != ""]

# Remove reviews without a valid product ID
reviews = reviews[reviews["product_id"] != ""]

# Convert numeric columns
reviews["rating"] = pd.to_numeric(
    reviews["rating"], errors="coerce"
)

reviews["price_usd"] = pd.to_numeric(
    reviews["price_usd"], errors="coerce"
)

# Retain only reviews connected to our product catalogue
product_ids = set(products["product_id"])

reviews = reviews[
    reviews["product_id"].isin(product_ids)
]

# Save cleaned reviews
reviews.to_csv(
    OUTPUT_DIR / "reviews_clean.csv",
    index=False
)


# ==========================================
# 3. FINAL SUMMARY
# ==========================================

print("\n========== CLEANING COMPLETED ==========")

print("Products remaining:", len(products))
print("Reviews remaining:", len(reviews))

print("\nProduct categories:")
print(products["primary_category"].value_counts())

print("\nReview skin types:")
print(reviews["skin_type"].value_counts(dropna=False))

print("\nOutput files:")
print(OUTPUT_DIR / "products_clean.csv")
print(OUTPUT_DIR / "reviews_clean.csv")
