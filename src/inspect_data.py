
import pandas as pd
from pathlib import Path

# Locate project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
product_path = BASE_DIR / "data/raw/product_info.csv"
review_path = BASE_DIR / "data/raw/reviews_1250-end.csv"


def inspect_dataset(file_path, name):

    print("\n" + "=" * 60)
    print(f"DATASET: {name}")
    print("=" * 60)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None

    df = pd.read_csv(file_path, low_memory=False)

    # Dataset dimensions
    print("\nShape (rows, columns):")
    print(df.shape)

    # Column names
    print("\nColumns:")
    print(df.columns.tolist())

    # Data types
    print("\nData types:")
    print(df.dtypes)

    # Missing values
    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(15))

    # Duplicate rows
    print("\nDuplicate rows:", df.duplicated().sum())

    # Sample records
    print("\nFirst 2 records:")
    print(df.head(2).to_string(index=False, max_cols=10,
                               max_colwidth=60))

    return df


# Inspect both datasets
products = inspect_dataset(product_path, "PRODUCTS")
reviews = inspect_dataset(review_path, "REVIEWS")


# Check whether products and reviews can be connected
if products is not None and reviews is not None:
    if "product_id" in products.columns and "product_id" in reviews.columns:

        product_ids = set(
            products["product_id"].dropna().astype(str).str.strip()
        )

        review_ids = reviews["product_id"].dropna().astype(str).str.strip()

        matching = review_ids.isin(product_ids).sum()

        print("\n" + "=" * 60)
        print("PRODUCT-REVIEW MATCHING")
        print("=" * 60)

        print("Unique products:", len(product_ids))
        print("Matching review rows:", matching)
        print("Unmatched review rows:", len(reviews) - matching)
    else:
        print("\nproduct_id column missing. Check the column names.")

print("\n========== PRODUCT CATEGORIES ==========")

print(products["primary_category"].value_counts(dropna=False))

print("\n========== SECONDARY CATEGORIES ==========")

print(products["secondary_category"].value_counts().head(20))

print("\n========== REVIEW SKIN TYPES ==========")

print(reviews["skin_type"].value_counts(dropna=False))