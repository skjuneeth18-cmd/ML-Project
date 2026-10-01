import pandas as pd
from pathlib import Path


# =========================================================
# WATER QUALITY CLASSIFICATION
# 01 - LOAD AND INSPECT DATASET
# =========================================================

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATASET_PATH = BASE_DIR / "Dataset" / "water_Quality.csv"


# ---------------------------------------------------------
# 1. Check whether dataset exists
# ---------------------------------------------------------

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at:\n{DATASET_PATH}"
    )


# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATASET_PATH)


print("=" * 65)
print("WATER QUALITY CLASSIFICATION - DATASET INSPECTION")
print("=" * 65)


# ---------------------------------------------------------
# 3. Basic information
# ---------------------------------------------------------

print("\n1. DATASET SHAPE")
print("-" * 65)
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ---------------------------------------------------------
# 4. First five rows
# ---------------------------------------------------------

print("\n2. FIRST 5 ROWS")
print("-" * 65)
print(df.head())


# ---------------------------------------------------------
# 5. Column names
# ---------------------------------------------------------

print("\n3. COLUMN NAMES")
print("-" * 65)

for number, column in enumerate(df.columns, start=1):
    print(f"{number:2}. {column}")


# ---------------------------------------------------------
# 6. Data types
# ---------------------------------------------------------

print("\n4. DATA TYPES")
print("-" * 65)
print(df.dtypes)


# ---------------------------------------------------------
# 7. Dataset information
# ---------------------------------------------------------

print("\n5. DATASET INFORMATION")
print("-" * 65)
df.info()


# ---------------------------------------------------------
# 8. Missing values
# ---------------------------------------------------------

missing_values = df.isnull().sum()

print("\n6. MISSING VALUES BY COLUMN")
print("-" * 65)
print(missing_values)

print("\nTotal Missing Values:", missing_values.sum())


# ---------------------------------------------------------
# 9. Duplicate rows
# ---------------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\n7. DUPLICATE ROWS")
print("-" * 65)
print("Duplicate Rows:", duplicate_count)


# ---------------------------------------------------------
# 10. Target distribution
# ---------------------------------------------------------

if "is_safe" not in df.columns:
    raise ValueError(
        "Target column 'is_safe' was not found in the dataset."
    )


print("\n8. TARGET DISTRIBUTION")
print("-" * 65)

target_counts = df["is_safe"].value_counts().sort_index()

print(target_counts)


print("\nTarget Percentage:")
print(
    (df["is_safe"].value_counts(normalize=True)
     .sort_index() * 100).round(2)
)


# ---------------------------------------------------------
# 11. Statistical summary
# ---------------------------------------------------------

print("\n9. STATISTICAL SUMMARY")
print("-" * 65)
print(df.describe())


# ---------------------------------------------------------
# 12. Final message
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("DATASET INSPECTION COMPLETED SUCCESSFULLY")
print("=" * 65)