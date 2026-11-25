
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

# Load cleaned datasets
products = pd.read_csv(DATA_DIR / "products_clean.csv")
reviews = pd.read_csv(DATA_DIR / "reviews_clean.csv")

# Standardize product IDs
products["product_id"] = products["product_id"].astype("string").str.strip()
reviews["product_id"] = reviews["product_id"].astype("string").str.strip()

# Count reviews for each product
review_counts = (
    reviews.groupby("product_id")
    .size()
    .reset_index(name="review_count")
)

# Merge counts with the complete product catalogue
result = products.merge(
    review_counts,
    on="product_id",
    how="left"
)

# Products without reviews get a count of zero
result["review_count"] = result["review_count"].fillna(0).astype(int)

# Summary statistics
total_products = len(result)
products_with_reviews = (result["review_count"] > 0).sum()
products_without_reviews = (result["review_count"] == 0).sum()

print("\n========== REVIEW COVERAGE ==========")
print("Total products:", total_products)
print("Products with reviews:", products_with_reviews)
print("Products without reviews:", products_without_reviews)

print("\n========== REVIEW STATISTICS ==========")
print("Total reviews:", result["review_count"].sum())
print("Average reviews per product:", round(result["review_count"].mean(), 2))
print("Median reviews per product:", result["review_count"].median())
print("Maximum reviews for one product:", result["review_count"].max())

# Distribution of review counts
print("\n========== REVIEW COUNT DISTRIBUTION ==========")

bins = [-1, 0, 1, 5, 10, 50, 100, float("inf")]
labels = [
    "0 reviews",
    "1 review",
    "2-5 reviews",
    "6-10 reviews",
    "11-50 reviews",
    "51-100 reviews",
    "More than 100 reviews"
]

result["review_group"] = pd.cut(
    result["review_count"],
    bins=bins,
    labels=labels
)

print(result["review_group"].value_counts(sort=False))

# Top 10 products by review count
print("\n========== TOP 10 REVIEWED PRODUCTS ==========")

name_column = (
    "product_name" if "product_name" in result.columns
    else "product_id"
)

print(
    result.nlargest(10, "review_count")[
        ["product_id", name_column, "review_count"]
    ].to_string(index=False)
)

# Save counts for every product
output_path = DATA_DIR / "product_review_counts.csv"

result.to_csv(output_path, index=False)

print("\nSaved review counts to:", output_path)
