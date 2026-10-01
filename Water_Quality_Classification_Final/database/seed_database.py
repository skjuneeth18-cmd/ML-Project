"""
WATER QUALITY CLASSIFICATION
Database seeding script.

Builds database/water_quality.db from database/schema.sql, then
fills it with:
  - water_samples   <- data/water_quality.csv     (1000 rows)
  - model_metrics   <- data/model_results.csv

Run this once from the project root (or again, any time you
want to reset the database to a clean state):

    python database/seed_database.py
"""

import sqlite3
import csv
from pathlib import Path

# ---------------------------------------------------------
# Paths (all relative to the project root, one level up
# from this file)
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "water_quality.db"
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"
DATASET_PATH = BASE_DIR / "data" / "water_quality.csv"
METRICS_PATH = BASE_DIR / "data" / "model_results.csv"

# The 20 input parameters, in the exact order the models expect
FEATURES = [
    "aluminium", "ammonia", "arsenic", "barium", "cadmium",
    "chloramine", "chromium", "copper", "flouride", "bacteria",
    "viruses", "lead", "nitrates", "nitrites", "mercury",
    "perchlorate", "radium", "selenium", "silver", "uranium",
]


def create_schema(conn):
    """Run schema.sql to (re)create all tables."""
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())


def seed_water_samples(conn):
    """Load the training dataset into the water_samples table."""
    with open(DATASET_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            try:
                values = [float(row[feat]) for feat in FEATURES]
                is_safe = int(float(row["is_safe"]))
            except (ValueError, KeyError):
                # skip malformed rows (some public copies of this
                # dataset contain a few '#NUM!' cells)
                continue
            rows.append(values + [is_safe])

    placeholders = ", ".join(["?"] * (len(FEATURES) + 1))
    columns = ", ".join(FEATURES + ["is_safe"])
    conn.executemany(
        f"INSERT INTO water_samples ({columns}) VALUES ({placeholders})",
        rows,
    )
    print(f"Inserted {len(rows)} rows into water_samples")


def seed_model_metrics(conn):
    """Load the model evaluation results into model_metrics."""
    if not METRICS_PATH.exists():
        print("model_results.csv not found, skipping model_metrics seed.")
        return

    with open(METRICS_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            (
                row["Model"],
                float(row["Accuracy"]),
                float(row["Precision"]),
                float(row["Recall"]),
                float(row["F1 Score"]),
            )
            for row in reader
        ]

    conn.executemany(
        "INSERT INTO model_metrics (model, accuracy, precision_score, recall, f1_score) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    print(f"Inserted {len(rows)} rows into model_metrics")


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        create_schema(conn)
        seed_water_samples(conn)
        seed_model_metrics(conn)
        conn.commit()
        print(f"\nDatabase created at: {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
