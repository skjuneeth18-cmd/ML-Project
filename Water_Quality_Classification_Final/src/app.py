import streamlit as st
import pandas as pd
import joblib

from pathlib import Path


# =========================================================
# WATER QUALITY CLASSIFICATION
# STREAMLIT APPLICATION
# =========================================================

# ---------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

RESULTS_DIR = BASE_DIR / "results"

METRIC_PATH = (
    RESULTS_DIR / "metrics" / "model_results.csv"
)


# ---------------------------------------------------------
# 2. Feature names
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
# 3. Load models
# ---------------------------------------------------------

@st.cache_resource
def load_models():

    svm_model = joblib.load(
        MODEL_DIR / "svm_model.pkl"
    )

    rf_model = joblib.load(
        MODEL_DIR / "random_forest_model.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "scaler.pkl"
    )

    return (
        svm_model,
        rf_model,
        scaler
    )


# ---------------------------------------------------------
# 4. Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Water Quality Classification",
    page_icon="💧",
    layout="wide"
)


# ---------------------------------------------------------
# 5. Load models safely
# ---------------------------------------------------------

try:

    (
        svm_model,
        rf_model,
        scaler
    ) = load_models()

except Exception as error:

    st.error(
        "Model files could not be loaded."
    )

    st.info(
        "Run Steps 04, 05 and 06 before starting "
        "the Streamlit application."
    )

    st.stop()


# ---------------------------------------------------------
# 6. Header
# ---------------------------------------------------------

st.title(
    "💧 Water Quality Classification"
)

st.subheader(
    "SVM and Random Forest Based Classification"
)

st.write(
    "Enter the water-quality parameters below "
    "to obtain predictions from two machine-learning models."
)


st.divider()


# ---------------------------------------------------------
# 7. Project information
# ---------------------------------------------------------

with st.expander(
    "ℹ️ About This Project"
):

    st.write(
        "This project uses supervised machine learning "
        "to classify water samples into the safety classes "
        "represented by the dataset target variable `is_safe`."
    )

    st.write(
        "**Algorithms:** Support Vector Machine (SVM) "
        "and Random Forest."
    )

    st.write(
        "**Evaluation:** Accuracy, Precision, Recall "
        "and F1 Score."
    )


# ---------------------------------------------------------
# 8. Input section
# ---------------------------------------------------------

st.header(
    "🧪 Water Quality Parameters"
)

st.caption(
    "Enter values using the same units and scale as "
    "the training dataset."
)


input_values = {}


col1, col2 = st.columns(2)


for index, feature in enumerate(FEATURES):

    if index % 2 == 0:

        with col1:

            input_values[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
                key=feature
            )

    else:

        with col2:

            input_values[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
                key=feature
            )


# ---------------------------------------------------------
# 9. Buttons
# ---------------------------------------------------------

predict_button = st.button(
    "🔍 Predict Water Quality",
    use_container_width=True
)


reset_button = st.button(
    "🔄 Reset Inputs"
)


if reset_button:

    st.rerun()


# ---------------------------------------------------------
# 10. Prediction
# ---------------------------------------------------------

if predict_button:

    input_data = pd.DataFrame(
        [input_values],
        columns=FEATURES
    )


    # -----------------------------------------------------
    # SVM
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Random Forest
    # -----------------------------------------------------

    rf_prediction = rf_model.predict(
        input_data
    )[0]

    rf_probability = rf_model.predict_proba(
        input_data
    )[0]

    rf_confidence = (
        max(rf_probability) * 100
    )


    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    st.divider()

    st.header(
        "📊 Prediction Results"
    )


    result_col1, result_col2 = st.columns(2)


    # SVM
    with result_col1:

        st.subheader("SVM")

        if svm_prediction == 1:

            st.success(
                "Classification: SAFE"
            )

        else:

            st.error(
                "Classification: UNSAFE"
            )

        st.metric(
            "Model Probability",
            f"{svm_confidence:.2f}%"
        )


    # Random Forest
    with result_col2:

        st.subheader("Random Forest")

        if rf_prediction == 1:

            st.success(
                "Classification: SAFE"
            )

        else:

            st.error(
                "Classification: UNSAFE"
            )

        st.metric(
            "Model Probability",
            f"{rf_confidence:.2f}%"
        )


    # -----------------------------------------------------
    # Agreement
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "🤝 Model Agreement"
    )


    if svm_prediction == rf_prediction:

        if svm_prediction == 1:

            st.success(
                "Both models classify the sample as SAFE."
            )

        else:

            st.error(
                "Both models classify the sample as UNSAFE."
            )

    else:

        st.warning(
            "The two models produced different classifications."
        )


    # -----------------------------------------------------
    # Probability comparison
    # -----------------------------------------------------

    st.subheader(
        "📈 Prediction Probability"
    )


    probability_table = pd.DataFrame({

        "Model": [
            "SVM",
            "Random Forest"
        ],

        "Prediction": [
            "SAFE" if svm_prediction == 1
            else "UNSAFE",

            "SAFE" if rf_prediction == 1
            else "UNSAFE"
        ],

        "Probability (%)": [
            svm_confidence,
            rf_confidence
        ]
    })


    st.dataframe(
        probability_table,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# 11. Model performance
# ---------------------------------------------------------

st.divider()

st.header(
    "🏆 Model Performance"
)


if METRIC_PATH.exists():

    results = pd.read_csv(
        METRIC_PATH
    )

    display_results = results.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        display_results[column] = (
            display_results[column] * 100
        ).round(2)


    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    best_model = results.loc[
        results["F1 Score"].idxmax(),
        "Model"
    ]

    best_f1 = results["F1 Score"].max()

    st.info(
        f"Best model based on F1 Score: "
        f"**{best_model}** "
        f"({best_f1:.4f})"
    )

else:

    st.warning(
        "Model evaluation results are not available yet. "
        "Run 07_evaluate_models.py."
    )


# ---------------------------------------------------------
# 12. Disclaimer
# ---------------------------------------------------------

st.divider()

st.warning(
    "Important: This application provides a machine-learning "
    "classification based on the training dataset. The "
    "prediction probability is not a guarantee of actual "
    "water safety and should not replace laboratory testing."
)


# ---------------------------------------------------------
# 13. Footer
# ---------------------------------------------------------

st.caption(
    "Water Quality Classification Using SVM and Random Forest"
)