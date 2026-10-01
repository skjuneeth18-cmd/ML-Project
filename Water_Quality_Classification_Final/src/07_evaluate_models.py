import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# WATER QUALITY CLASSIFICATION
# 07 - MODEL EVALUATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

RESULTS_DIR = BASE_DIR / "results"

GRAPH_DIR = RESULTS_DIR / "graphs"

METRIC_DIR = RESULTS_DIR / "metrics"


GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRIC_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# 1. Load test data
# ---------------------------------------------------------

X_test_scaled = pd.read_csv(
    MODEL_DIR / "X_test_scaled.csv"
)

X_test = pd.read_csv(
    MODEL_DIR / "X_test.csv"
)

y_test = pd.read_csv(
    MODEL_DIR / "y_test.csv"
).squeeze()


# ---------------------------------------------------------
# 2. Load models
# ---------------------------------------------------------

svm_model = joblib.load(
    MODEL_DIR / "svm_model.pkl"
)

rf_model = joblib.load(
    MODEL_DIR / "random_forest_model.pkl"
)


print("=" * 65)
print("MODEL EVALUATION")
print("=" * 65)


# ---------------------------------------------------------
# 3. Predictions
# ---------------------------------------------------------

svm_predictions = svm_model.predict(
    X_test_scaled
)

rf_predictions = rf_model.predict(
    X_test
)


# ---------------------------------------------------------
# 4. Metric function
# ---------------------------------------------------------

def calculate_metrics(y_true, predictions):

    return {
        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_true,
            predictions,
            zero_division=0
        )
    }


svm_metrics = calculate_metrics(
    y_test,
    svm_predictions
)

rf_metrics = calculate_metrics(
    y_test,
    rf_predictions
)


# ---------------------------------------------------------
# 5. Print SVM results
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("SVM RESULTS")
print("=" * 65)

for metric, value in svm_metrics.items():
    print(f"{metric:<12}: {value:.4f}")


print("\nSVM Classification Report:")
print(
    classification_report(
        y_test,
        svm_predictions,
        zero_division=0
    )
)


# ---------------------------------------------------------
# 6. Print Random Forest results
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("RANDOM FOREST RESULTS")
print("=" * 65)

for metric, value in rf_metrics.items():
    print(f"{metric:<12}: {value:.4f}")


print("\nRandom Forest Classification Report:")
print(
    classification_report(
        y_test,
        rf_predictions,
        zero_division=0
    )
)


# ---------------------------------------------------------
# 7. Confusion matrix function
# ---------------------------------------------------------

def save_confusion_matrix(
    y_true,
    predictions,
    title,
    filename
):

    cm = confusion_matrix(
        y_true,
        predictions
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(title)

    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["Unsafe (0)", "Safe (1)"]
    )

    plt.yticks(
        [0, 1],
        ["Unsafe (0)", "Safe (1)"]
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    for row in range(cm.shape[0]):

        for column in range(cm.shape[1]):

            plt.text(
                column,
                row,
                str(cm[row, column]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / filename,
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# 8. Save confusion matrices
# ---------------------------------------------------------

save_confusion_matrix(
    y_test,
    svm_predictions,
    "SVM Confusion Matrix",
    "svm_confusion_matrix.png"
)

save_confusion_matrix(
    y_test,
    rf_predictions,
    "Random Forest Confusion Matrix",
    "rf_confusion_matrix.png"
)


# ---------------------------------------------------------
# 9. Save results
# ---------------------------------------------------------

results = pd.DataFrame([
    {
        "Model": "SVM",
        **svm_metrics
    },
    {
        "Model": "Random Forest",
        **rf_metrics
    }
])


RESULT_PATH = (
    METRIC_DIR / "model_results.csv"
)

results.to_csv(
    RESULT_PATH,
    index=False
)


print("\nRESULT FILE SAVED:")
print(RESULT_PATH)

print("\nCONFUSION MATRIX GRAPHS SAVED:")
print(GRAPH_DIR)


print("\n" + "=" * 65)
print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 65)