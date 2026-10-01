import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# =========================================================
# WATER QUALITY CLASSIFICATION
# 03 - EXPLORATORY DATA ANALYSIS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR / "Dataset" / "cleaned_water_quality.csv"
)

GRAPH_DIR = BASE_DIR / "results" / "graphs"

GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Feature list
# ---------------------------------------------------------

FEATURES = [
    "aluminium",
    "ammonia",
    "arsenic",
    "barium",
    "cadmium",
    "chloramine",
    "chromium",
    "copper",
    "flouride",
    "bacteria",
    "viruses",
    "lead",
    "nitrates",
    "nitrites",
    "mercury",
    "perchlorate",
    "radium",
    "selenium",
    "silver",
    "uranium"
]


# ---------------------------------------------------------
# 1. Load cleaned dataset
# ---------------------------------------------------------

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found at:\n{DATASET_PATH}\n"
        "Run 02_data_cleaning.py first."
    )


df = pd.read_csv(DATASET_PATH)


# ---------------------------------------------------------
# 2. Validate columns
# ---------------------------------------------------------

required_columns = FEATURES + ["is_safe"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )


print("=" * 65)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 65)


# ---------------------------------------------------------
# 3. Dataset summary
# ---------------------------------------------------------

print("\n1. DATASET SHAPE")
print("-" * 65)
print(df.shape)


print("\n2. STATISTICAL SUMMARY")
print("-" * 65)
print(df[FEATURES].describe())


# ---------------------------------------------------------
# 4. Target distribution
# ---------------------------------------------------------

target_counts = (
    df["is_safe"]
    .value_counts()
    .sort_index()
)

print("\n3. TARGET DISTRIBUTION")
print("-" * 65)
print(target_counts)


# ---------------------------------------------------------
# 5. Target class graph
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    ["Unsafe (0)", "Safe (1)"],
    [
        target_counts.get(0, 0),
        target_counts.get(1, 0)
    ]
)

plt.title("Water Safety Class Distribution")
plt.xlabel("Safety Class")
plt.ylabel("Number of Samples")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "class_distribution.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 6. Feature distributions
# ---------------------------------------------------------

fig, axes = plt.subplots(
    5,
    4,
    figsize=(16, 18)
)

axes = axes.flatten()

for index, feature in enumerate(FEATURES):

    axes[index].hist(
        df[feature],
        bins=30
    )

    axes[index].set_title(feature)
    axes[index].set_xlabel("Value")
    axes[index].set_ylabel("Frequency")


plt.suptitle(
    "Distributions of Water Quality Parameters",
    fontsize=16
)

plt.tight_layout(
    rect=[0, 0, 1, 0.97]
)

plt.savefig(
    GRAPH_DIR / "feature_distributions.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 7. Box plots
# ---------------------------------------------------------

plt.figure(figsize=(16, 7))

df[FEATURES].boxplot()

plt.title("Box Plot of Water Quality Parameters")
plt.xlabel("Parameters")
plt.ylabel("Values")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "feature_boxplots.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 8. Correlation matrix
# ---------------------------------------------------------

correlation = df[FEATURES + ["is_safe"]].corr()


print("\n4. CORRELATION WITH TARGET")
print("-" * 65)

target_correlation = (
    correlation["is_safe"]
    .drop("is_safe")
    .sort_values(
        key=lambda values: values.abs(),
        ascending=False
    )
)

print(target_correlation)


# ---------------------------------------------------------
# 9. Correlation heatmap
# ---------------------------------------------------------

plt.figure(figsize=(16, 13))

plt.imshow(
    correlation,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar(
    label="Correlation"
)

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "correlation_heatmap.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 10. Target correlation graph
# ---------------------------------------------------------

plt.figure(figsize=(10, 7))

plt.barh(
    target_correlation.index,
    target_correlation.values
)

plt.xlabel("Correlation with is_safe")
plt.ylabel("Feature")

plt.title(
    "Feature Correlation with Water Safety Class"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "target_correlation.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 11. Final message
# ---------------------------------------------------------

print("\n5. GRAPHS CREATED")
print("-" * 65)

print("Class distribution")
print("Feature distributions")
print("Feature box plots")
print("Correlation heatmap")
print("Target correlation graph")


print("\nGraphs saved in:")
print(GRAPH_DIR)


print("\n" + "=" * 65)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 65)