"""
WATER QUALITY CLASSIFICATION
Flask REST API backend.
=====================================================

What this file does:
  1. Loads the three pre-trained model files from ../models/
     (svm_model.pkl, random_forest_model.pkl, scaler.pkl)
  2. Exposes a REST API (the /api/... routes below) that the
     frontend calls with fetch()
  3. Also serves the frontend's static files (index.html,
     style.css, script.js), so the whole website runs from a
     single command: `python backend/app.py`
  4. Reads and writes prediction history to the SQLite database
     in ../database/water_quality.db

Run:
    pip install -r backend/requirements.txt
    python database/seed_database.py    # once, to build the DB
    python backend/app.py

Then open:
    http://localhost:5000
"""

import sqlite3
import math
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g, send_from_directory
from flask_cors import CORS

# ---------------------------------------------------------
# 1. Paths — everything is relative to the project root,
#    which is one level up from this backend/ folder
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "database" / "water_quality.db"
FRONTEND_DIR = BASE_DIR / "frontend"

# The 20 water-quality parameters, in the exact column order
# the models were trained on. Order matters for the Random
# Forest model, which receives a plain (unscaled) DataFrame.
FEATURES = [
    "aluminium", "ammonia", "arsenic", "barium", "cadmium",
    "chloramine", "chromium", "copper", "flouride", "bacteria",
    "viruses", "lead", "nitrates", "nitrites", "mercury",
    "perchlorate", "radium", "selenium", "silver", "uranium",
]

# Approximate guideline maximums, shown only in the frontend's
# form labels for context — the ML models make the actual call.
FEATURE_INFO = {
    "aluminium": {"unit": "mg/L", "safe_max": 2.8},
    "ammonia": {"unit": "mg/L", "safe_max": 32.5},
    "arsenic": {"unit": "mg/L", "safe_max": 0.01},
    "barium": {"unit": "mg/L", "safe_max": 2.0},
    "cadmium": {"unit": "mg/L", "safe_max": 0.005},
    "chloramine": {"unit": "mg/L", "safe_max": 4.0},
    "chromium": {"unit": "mg/L", "safe_max": 0.1},
    "copper": {"unit": "mg/L", "safe_max": 1.3},
    "flouride": {"unit": "mg/L", "safe_max": 1.5},
    "bacteria": {"unit": "CFU/mL", "safe_max": 0.0},
    "viruses": {"unit": "CFU/mL", "safe_max": 0.0},
    "lead": {"unit": "mg/L", "safe_max": 0.015},
    "nitrates": {"unit": "mg/L", "safe_max": 10.0},
    "nitrites": {"unit": "mg/L", "safe_max": 1.0},
    "mercury": {"unit": "mg/L", "safe_max": 0.002},
    "perchlorate": {"unit": "mg/L", "safe_max": 56.0},
    "radium": {"unit": "pCi/L", "safe_max": 5.0},
    "selenium": {"unit": "mg/L", "safe_max": 0.5},
    "silver": {"unit": "mg/L", "safe_max": 0.1},
    "uranium": {"unit": "mg/L", "safe_max": 0.3},
}

# ---------------------------------------------------------
# 2. Create the Flask app and load the trained models once,
#    at startup (not on every request — that would be slow)
# ---------------------------------------------------------

app = Flask(__name__)
CORS(app)  # allow the frontend to call this API from any origin

svm_model = joblib.load(MODEL_DIR / "svm_model.pkl")
rf_model = joblib.load(MODEL_DIR / "random_forest_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")


# ---------------------------------------------------------
# 3. Database helpers
#    Flask's `g` object holds one connection per request and
#    closes it automatically when the request finishes.
# ---------------------------------------------------------

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row  # rows behave like dicts
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


# ---------------------------------------------------------
# 4. Serve the static frontend
#    "/" returns index.html; any other path is looked up
#    inside the frontend/ folder (style.css, script.js, ...)
# ---------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ---------------------------------------------------------
# 5. GET /api/health — simple check that the server (and the
#    models) are up
# ---------------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": True,
        "time": datetime.now(timezone.utc).isoformat(),
    })


# ---------------------------------------------------------
# 6. GET /api/features — tells the frontend which 20 fields
#    to render in the prediction form, with their units
# ---------------------------------------------------------

@app.route("/api/features")
def features():
    return jsonify({"features": FEATURES, "info": FEATURE_INFO})


# ---------------------------------------------------------
# 7. POST /api/predict — the main endpoint.
#    Body: { "aluminium": 1.65, "ammonia": 9.08, ... }  (20 keys)
#    Returns SVM + Random Forest predictions, and logs the
#    result to the predictions table.
# ---------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body must be an object"}), 400

    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        values = {f: float(payload[f]) for f in FEATURES}
    except (TypeError, ValueError):
        return jsonify({"error": "All feature values must be numeric"}), 400

    if not all(math.isfinite(value) for value in values.values()):
        return jsonify({"error": "All feature values must be finite numbers"}), 400

    if any(value < 0 for value in values.values()):
        return jsonify({"error": "Feature values cannot be negative"}), 400

    # Build a single-row DataFrame with columns in the exact
    # order the models expect.
    input_data = pd.DataFrame([values], columns=FEATURES)

    # --- SVM: needs the StandardScaler-transformed input -----
    input_scaled = pd.DataFrame(
        scaler.transform(input_data),
        columns=FEATURES,
    )
    svm_pred = int(svm_model.predict(input_scaled)[0])
    svm_conf = float(max(svm_model.predict_proba(input_scaled)[0]) * 100)

    # --- Random Forest: trained on the raw (unscaled) input ---
    rf_pred = int(rf_model.predict(input_data)[0])
    rf_conf = float(max(rf_model.predict_proba(input_data)[0]) * 100)

    agree = svm_pred == rf_pred
    # If the two models disagree, fall back to the Random Forest
    # result (it scored a slightly higher F1 in data/model_results.csv).
    final_pred = svm_pred if agree else rf_pred

    result = {
        "input": values,
        "svm": {
            "prediction": svm_pred,
            "label": "SAFE" if svm_pred == 1 else "UNSAFE",
            "confidence": round(svm_conf, 2),
        },
        "random_forest": {
            "prediction": rf_pred,
            "label": "SAFE" if rf_pred == 1 else "UNSAFE",
            "confidence": round(rf_conf, 2),
        },
        "agreement": agree,
        "final_prediction": final_pred,
        "final_label": "SAFE" if final_pred == 1 else "UNSAFE",
    }

    # Log this prediction to the database so it shows up in the
    # site's "Recent predictions" history table.
    db = get_db()
    columns = FEATURES + [
        "svm_prediction", "svm_confidence",
        "rf_prediction", "rf_confidence",
        "final_prediction", "models_agree",
    ]
    row_values = [values[f] for f in FEATURES] + [
        svm_pred, svm_conf, rf_pred, rf_conf, final_pred, int(agree),
    ]
    placeholders = ", ".join(["?"] * len(columns))
    db.execute(
        f"INSERT INTO predictions ({', '.join(columns)}) VALUES ({placeholders})",
        row_values,
    )
    db.commit()

    return jsonify(result)


# ---------------------------------------------------------
# 8. GET /api/history — recent predictions, newest first
#    Query param: ?limit=20 (default 20, max 200)
# ---------------------------------------------------------

@app.route("/api/history")
def history():
    limit = request.args.get("limit", default=20, type=int)
    limit = max(1, min(limit, 200))

    db = get_db()
    rows = db.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()

    return jsonify([dict(row) for row in rows])


# ---------------------------------------------------------
# 9. GET /api/metrics — SVM vs Random Forest evaluation scores
# ---------------------------------------------------------

@app.route("/api/metrics")
def metrics():
    db = get_db()
    rows = db.execute("SELECT * FROM model_metrics").fetchall()
    return jsonify([dict(row) for row in rows])


# ---------------------------------------------------------
# 10. GET /api/stats — dataset-wide numbers for the dashboard
#     (safe/unsafe split, average of each parameter)
# ---------------------------------------------------------

@app.route("/api/stats")
def stats():
    db = get_db()

    total = db.execute("SELECT COUNT(*) AS c FROM water_samples").fetchone()["c"]
    safe = db.execute(
        "SELECT COUNT(*) AS c FROM water_samples WHERE is_safe = 1"
    ).fetchone()["c"]
    unsafe = total - safe

    avg_row = db.execute(
        f"SELECT {', '.join(f'AVG({f}) AS {f}' for f in FEATURES)} FROM water_samples"
    ).fetchone()

    total_predictions = db.execute(
        "SELECT COUNT(*) AS c FROM predictions"
    ).fetchone()["c"]

    return jsonify({
        "dataset": {
            "total_samples": total,
            "safe_count": safe,
            "unsafe_count": unsafe,
            "safe_ratio": round(safe / total, 4) if total else 0,
            "feature_averages": {f: round(avg_row[f], 4) for f in FEATURES},
        },
        "site_predictions_made": total_predictions,
    })


# ---------------------------------------------------------
# 11. GET /api/samples — a few random rows from the dataset,
#     used by the "Load a random sample" button on the form
# ---------------------------------------------------------

@app.route("/api/samples")
def samples():
    n = request.args.get("n", default=1, type=int)
    n = max(1, min(n, 20))

    db = get_db()
    rows = db.execute(
        "SELECT * FROM water_samples ORDER BY RANDOM() LIMIT ?", (n,)
    ).fetchall()

    return jsonify([dict(row) for row in rows])


# ---------------------------------------------------------
# 12. Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    if not DB_PATH.exists():
        print(
            "WARNING: database/water_quality.db not found.\n"
            "Run 'python database/seed_database.py' first."
        )
    app.run(debug=False, host="127.0.0.1", port=5000)
