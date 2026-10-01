import pandas as pd
import joblib

from pathlib import Path

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# =========================================================
# WATER QUALITY CLASSIFICATION
# 05 - SVM TRAINING
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

X_TRAIN_PATH = MODEL_DIR / "X_train_scaled.csv"
Y_TRAIN_PATH = MODEL_DIR / "y_train.csv"

MODEL_PATH = MODEL_DIR / "svm_model.pkl"


# ---------------------------------------------------------
# 1. Load training data
# ---------------------------------------------------------

X_train = pd.read_csv(X_TRAIN_PATH)

y_train = pd.read_csv(
    Y_TRAIN_PATH
).squeeze()


if X_train.empty or y_train.empty:
    raise ValueError(
        "Training data is empty."
    )


print("=" * 65)
print("SUPPORT VECTOR MACHINE - MODEL TRAINING")
print("=" * 65)


print("\n1. TRAINING DATA")
print("-" * 65)
print("Features:", X_train.shape)
print("Target  :", y_train.shape)


# ---------------------------------------------------------
# 2. Create SVM model
# ---------------------------------------------------------

svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)


# ---------------------------------------------------------
# 3. Cross-validation
# ---------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    svm_model,
    X_train,
    y_train,
    cv=cv,
    scoring="f1"
)


print("\n2. 5-FOLD CROSS-VALIDATION")
print("-" * 65)

print(
    "Fold F1 Scores:",
    [round(score, 4) for score in cv_scores]
)

print(
    f"Mean F1 Score: {cv_scores.mean():.4f}"
)

print(
    f"Std F1 Score : {cv_scores.std():.4f}"
)


# ---------------------------------------------------------
# 4. Train final SVM
# ---------------------------------------------------------

svm_model.fit(
    X_train,
    y_train
)


print("\n3. FINAL SVM TRAINING")
print("-" * 65)
print("Kernel : RBF")
print("C      : 1.0")
print("Gamma  : scale")


# ---------------------------------------------------------
# 5. Save model
# ---------------------------------------------------------

joblib.dump(
    svm_model,
    MODEL_PATH
)


print("\n4. MODEL SAVED")
print("-" * 65)
print(MODEL_PATH)


print("\n" + "=" * 65)
print("SVM TRAINING COMPLETED SUCCESSFULLY")
print("=" * 65)