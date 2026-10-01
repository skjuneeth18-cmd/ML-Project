import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# =========================================================
# WATER QUALITY CLASSIFICATION
# 08 - MODEL COMPARISON
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

METRIC_PATH = (
    RESULTS_DIR / "metrics" / "model_results.csv"
)

GRAPH_DIR = RESULTS_DIR / "graphs"

GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# 1. Load results
# ---------------------------------------------------------

if not METRIC_PATH.exists():
    raise FileNotFoundError(
        "model_results.csv not found. "
        "Run 07_evaluate_models.py first."
    )


results = pd.read_csv(
    METRIC_PATH
)


print("=" * 65)
print("SVM VS RANDOM FOREST COMPARISON")
print("=" * 65)


print("\nMODEL PERFORMANCE")
print("-" * 65)
print(
    results.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 2. Find best model
# ---------------------------------------------------------

best_index = results[
    "F1 Score"
].idxmax()

best_model = results.loc[
    best_index,
    "Model"
]

best_f1 = results.loc[
    best_index,
    "F1 Score"
]


print("\nBEST MODEL")
print("-" * 65)

print("Model    :", best_model)
print(f"F1 Score : {best_f1:.4f}")


# ---------------------------------------------------------
# 3. Create comparison graph
# ---------------------------------------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

models = results["Model"].tolist()

x_positions = list(
    range(len(models))
)

width = 0.18


plt.figure(figsize=(11, 7))


for index, metric in enumerate(metrics):

    values = results[metric].tolist()

    positions = [
        x + (index - 1.5) * width
        for x in x_positions
    ]

    bars = plt.bar(
        positions,
        values,
        width=width,
        label=metric
    )

    # Display value above each bar
    for bar, value in zip(
        bars,
        values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.015,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )


plt.xticks(
    x_positions,
    models
)

plt.xlabel("Machine Learning Model")
plt.ylabel("Score")

plt.title(
    "Performance Comparison: SVM vs Random Forest"
)

plt.ylim(
    0,
    1.10
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()


GRAPH_PATH = (
    GRAPH_DIR / "model_comparison.png"
)


plt.savefig(
    GRAPH_PATH,
    dpi=300
)

plt.close()


print("\nCOMPARISON GRAPH SAVED:")
print(GRAPH_PATH)


print("\n" + "=" * 65)
print("MODEL COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 65)