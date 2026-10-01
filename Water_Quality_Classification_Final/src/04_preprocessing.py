import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================================================
# WATER QUALITY CLASSIFICATION
# 04 - DATA PREPROCESSING
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR / "Dataset" / "cleaned_water_quality.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. "
        "Run 02_data_cleaning.py first."
    )

df = pd.read_csv(DATASET_PATH)


# ---------------------------------------------------------
# 2. Separate features and target
# ---------------------------------------------------------

X = df.drop(
    columns=["is_safe"]
)

y = df["is_safe"]


print("=" * 65)
print("WATER QUALITY DATA PREPROCESSING")
print("=" * 65)


print("\n1. FEATURES")
print("-" * 65)
print("Number of features:", X.shape[1])


print("\n2. TARGET")
print("-" * 65)
print("Target column: is_safe")


# ---------------------------------------------------------
# 3. Train-test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n3. TRAIN-TEST SPLIT")
print("-" * 65)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ---------------------------------------------------------
# 4. Target distribution
# ---------------------------------------------------------

print("\n4. TRAINING TARGET DISTRIBUTION")
print("-" * 65)
print(y_train.value_counts().sort_index())


print("\nTESTING TARGET DISTRIBUTION")
print("-" * 65)
print(y_test.value_counts().sort_index())


# ---------------------------------------------------------
# 5. Feature scaling for SVM
# ---------------------------------------------------------

scaler = StandardScaler()

# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same fitted scaler
X_test_scaled = scaler.transform(X_test)


print("\n5. FEATURE SCALING")
print("-" * 65)
print("StandardScaler applied.")
print("Scaler fitted only on training data.")


# ---------------------------------------------------------
# 6. Save scaler
# ---------------------------------------------------------

scaler_path = MODEL_DIR / "scaler.pkl"

joblib.dump(
    scaler,
    scaler_path
)


# ---------------------------------------------------------
# 7. Save train/test datasets
# ---------------------------------------------------------

X_train.to_csv(
    MODEL_DIR / "X_train.csv",
    index=False
)

X_test.to_csv(
    MODEL_DIR / "X_test.csv",
    index=False
)

y_train.to_csv(
    MODEL_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    MODEL_DIR / "y_test.csv",
    index=False
)


# ---------------------------------------------------------
# 8. Save scaled data
# ---------------------------------------------------------

pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
).to_csv(
    MODEL_DIR / "X_train_scaled.csv",
    index=False
)


pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
).to_csv(
    MODEL_DIR / "X_test_scaled.csv",
    index=False
)


# ---------------------------------------------------------
# 9. Output
# ---------------------------------------------------------

print("\n6. OUTPUT FILES CREATED")
print("-" * 65)

print("scaler.pkl")
print("X_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")
print("X_train_scaled.csv")
print("X_test_scaled.csv")


print("\n" + "=" * 65)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 65)