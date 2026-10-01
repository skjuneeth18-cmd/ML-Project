import pandas as pd
import joblib

from pathlib import Path


# =========================================================
# WATER QUALITY CLASSIFICATION
# 09 - PREDICTION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"


# ---------------------------------------------------------
# Feature names
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
# 1. Check model files
# ---------------------------------------------------------

required_files = [
    MODEL_DIR / "svm_model.pkl",
    MODEL_DIR / "random_forest_model.pkl",
    MODEL_DIR / "scaler.pkl"
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required model file not found:\n{file_path}\n"
            "Run Steps 04, 05 and 06 first."
        )


# ---------------------------------------------------------
# 2. Load models
# ---------------------------------------------------------

svm_model = joblib.load(
    MODEL_DIR / "svm_model.pkl"
)

rf_model = joblib.load(
    MODEL_DIR / "random_forest_model.pkl"
)

scaler = joblib.load(
    MODEL_DIR / "scaler.pkl"
)


# ---------------------------------------------------------
# 3. Input
# ---------------------------------------------------------

print("=" * 65)
print("WATER QUALITY PREDICTION")
print("=" * 65)

print(
    "\nEnter one numerical value for each parameter."
)

print(
    "Use values in the same units and format as the dataset.\n"
)


input_values = []


for feature in FEATURES:

    while True:

        try:

            value = float(
                input(
                    f"Enter {feature}: "
                )
            )

            input_values.append(
                value
            )

            break

        except ValueError:

            print(
                "Invalid input. "
                "Please enter a numerical value."
            )


# ---------------------------------------------------------
# 4. Create DataFrame
# ---------------------------------------------------------

input_data = pd.DataFrame(
    [input_values],
    columns=FEATURES
)


# ---------------------------------------------------------
# 5. SVM prediction
# ---------------------------------------------------------

input_scaled = scaler.transform(
    input_data
)

svm_prediction = svm_model.predict(
    input_scaled
)[0]

svm_probability = svm_model.predict_proba(
    input_scaled
)[0]

svm_confidence = (
    max(svm_probability) * 100
)


# ---------------------------------------------------------
# 6. Random Forest prediction
# ---------------------------------------------------------

rf_prediction = rf_model.predict(
    input_data
)[0]

rf_probability = rf_model.predict_proba(
    input_data
)[0]

rf_confidence = (
    max(rf_probability) * 100
)


# ---------------------------------------------------------
# 7. Display results
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("PREDICTION RESULTS")
print("=" * 65)


print("\nSVM")
print("-" * 65)

if svm_prediction == 1:
    print("Classification : SAFE")
else:
    print("Classification : UNSAFE")

print(
    f"Model Probability: "
    f"{svm_confidence:.2f}%"
)


print("\nRANDOM FOREST")
print("-" * 65)

if rf_prediction == 1:
    print("Classification : SAFE")
else:
    print("Classification : UNSAFE")

print(
    f"Model Probability: "
    f"{rf_confidence:.2f}%"
)


# ---------------------------------------------------------
# 8. Model agreement
# ---------------------------------------------------------

print("\nMODEL AGREEMENT")
print("-" * 65)


if svm_prediction == rf_prediction:

    print(
        "Both models produced the same classification."
    )

else:

    print(
        "SVM and Random Forest produced "
        "different classifications."
    )


# ---------------------------------------------------------
# 9. Important interpretation
# ---------------------------------------------------------

print("\nIMPORTANT")
print("-" * 65)

print(
    "Model probability is an estimated probability "
    "from the classifier."
)

print(
    "It is not a guarantee of actual water safety."
)


print("\n" + "=" * 65)
print("PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 65)