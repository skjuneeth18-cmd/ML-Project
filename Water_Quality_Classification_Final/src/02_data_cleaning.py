import pandas as pd
from pathlib import Path


# =========================================================
# WATER QUALITY CLASSIFICATION
# 02 - DATA CLEANING
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "Dataset" / "water_Quality.csv"

OUTPUT_PATH = (
    BASE_DIR / "Dataset" / "cleaned_water_quality.csv"
)


# ---------------------------------------------------------
# 1. Check input file
# ---------------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Original dataset not found at:\n{INPUT_PATH}"
    )


# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------

df = pd.read_csv(INPUT_PATH)


print("=" * 65)
print("WATER QUALITY DATA CLEANING")
print("=" * 65)


# ---------------------------------------------------------
# 3. Original dataset information
# ---------------------------------------------------------

print("\n1. ORIGINAL DATASET")
print("-" * 65)

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ---------------------------------------------------------
# 4. Check missing values
# ---------------------------------------------------------

missing_before = df.isnull().sum().sum()

print("\n2. MISSING VALUES BEFORE CLEANING")
print("-" * 65)
print("Total Missing Values:", missing_before)


# ---------------------------------------------------------
# 5. Check duplicates
# ---------------------------------------------------------

duplicates_before = df.duplicated().sum()

print("\n3. DUPLICATES BEFORE CLEANING")
print("-" * 65)
print("Duplicate Rows:", duplicates_before)


# ---------------------------------------------------------
# 6. Remove duplicate rows
# ---------------------------------------------------------

if duplicates_before > 0:

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )


# ---------------------------------------------------------
# 7. Check missing values after cleaning
# ---------------------------------------------------------

missing_after = df.isnull().sum().sum()

duplicates_after = df.duplicated().sum()


# ---------------------------------------------------------
# 8. Display cleaning results
# ---------------------------------------------------------

print("\n4. CLEANING RESULTS")
print("-" * 65)

print("Original Rows       :", df.shape[0] + duplicates_before)
print("Rows Removed        :", duplicates_before)
print("Cleaned Rows        :", df.shape[0])
print("Columns             :", df.shape[1])

print("\nMissing Values After Cleaning:", missing_after)
print("Duplicate Rows After Cleaning:", duplicates_after)


# ---------------------------------------------------------
# 9. Validate target column
# ---------------------------------------------------------

if "is_safe" not in df.columns:
    raise ValueError(
        "Target column 'is_safe' was not found."
    )


# ---------------------------------------------------------
# 10. Save cleaned dataset
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n5. CLEANED DATASET SAVED")
print("-" * 65)
print(OUTPUT_PATH)


# ---------------------------------------------------------
# 11. Final target distribution
# ---------------------------------------------------------

print("\n6. CLEANED TARGET DISTRIBUTION")
print("-" * 65)
print(df["is_safe"].value_counts().sort_index())


print("\n" + "=" * 65)
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("=" * 65)