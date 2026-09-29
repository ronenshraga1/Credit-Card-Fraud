"""Streamlit demo for the project's saved fraud classifier."""

from collections import Counter
import csv
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "credit_card_fraud_model.joblib"
SAMPLE_PATH = ROOT / "sample_transactions.csv"
SCORE_NOTE = (
    "The fraud score is the model's classification score used for threshold-based "
    "decisions. It should not be interpreted as a calibrated real-world probability of fraud."
)
METADATA_COLUMNS = {"Class", "ActualClass", "ModelPrediction", "ModelScore", "ExampleType"}


@st.cache_resource(show_spinner="Loading the saved model...")
def load_model_package(path):
    package = joblib.load(path)
    if not isinstance(package, dict):
        raise ValueError("The model artifact must contain a model package dictionary.")
    missing = {"model", "threshold"} - package.keys()
    if missing:
        raise ValueError("Missing model package keys: " + ", ".join(sorted(missing)))
    model = package["model"]
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("The saved model does not support score-based inference.")
    threshold = float(package["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The saved decision threshold must be between 0 and 1.")

    # The original artifact stores the ordered schema on the fitted estimator.
    names = package.get("feature_names", getattr(model, "feature_names_in_", None))
    if names is None:
        raise ValueError("Missing feature_names and fitted model feature_names_in_.")
    names = list(names)
    if not names or not all(isinstance(name, str) for name in names):
        raise ValueError("The saved feature names must be a nonempty list of strings.")
    if len(set(names)) != len(names) or METADATA_COLUMNS.intersection(names):
        raise ValueError("The saved feature schema contains duplicates or label metadata.")
    if len(names) != getattr(model, "n_features_in_", len(names)):
        raise ValueError("The saved feature schema does not match the model's input count.")
    if hasattr(model, "feature_names_in_") and names != list(model.feature_names_in_):
        raise ValueError("The package feature order does not match the fitted model.")
    classes = np.asarray(getattr(model, "classes_", []))
    fraud_columns = np.flatnonzero(classes == 1)
    if len(classes) != 2 or len(fraud_columns) != 1 or 0 not in classes:
        raise ValueError("The saved model must have normal (0) and fraud (1) classes.")
    return model, threshold, names, int(fraud_columns[0])


def read_transactions(contents):
    text = contents.decode("utf-8-sig")
    header = next(csv.reader(StringIO(text)), [])
    duplicates = [name for name, count in Counter(header).items() if count > 1]
    if duplicates:
        raise ValueError("Duplicate CSV columns: " + ", ".join(duplicates))
    frame = pd.read_csv(StringIO(text))
    if frame.empty:
        raise ValueError("The CSV must contain at least one transaction.")
    return frame


@st.cache_data(show_spinner=False)
def load_samples(path):
    return read_transactions(path.read_bytes())


def model_inputs(frame, feature_names):
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError("Missing required features: " + ", ".join(missing))
    # Select and reorder strictly by the stored schema, excluding all metadata.
    inputs = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    invalid = [name for name in feature_names if not np.isfinite(inputs[name]).all()]
    if invalid:
        raise ValueError(
            "Features must contain finite numeric values; check: " + ", ".join(invalid)
        )
    return inputs


def predict_transactions(model, threshold, feature_names, fraud_column, frame):
    inputs = model_inputs(frame, feature_names)
    scores = model.predict_proba(inputs)[:, fraud_column]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("The model returned invalid fraud scores.")
    return scores, (scores >= threshold).astype(int)


def example_labels(samples):
    types = samples["ExampleType"].fillna("Sample Transaction").astype(str).tolist()
    counts = Counter(types)
    seen = Counter()
    labels = []
    for kind in types:
        seen[kind] += 1
        label = f"{kind} #{seen[kind]}" if counts[kind] > 1 else kind
        # Also avoid collisions with labels that already contain numbered suffixes.
        while label in labels:
            label += " (sample)"
        labels.append(label)
    return labels


def single_transaction_tab(model, threshold, feature_names, fraud_column):
    try:
        samples = load_samples(SAMPLE_PATH)
        model_inputs(samples, feature_names)
        missing = {"ActualClass", "ExampleType"} - set(samples.columns)
        if missing:
            raise ValueError("Missing sample metadata: " + ", ".join(sorted(missing)))
        if not samples["ActualClass"].isin([0, 1]).all():
            raise ValueError("Sample ActualClass values must be 0 or 1.")
    except Exception as exc:
        st.error(f"Could not load sample transactions: {exc}")
        return

    labels = example_labels(samples)
    selection = st.selectbox(
        "Choose a sample transaction", range(len(samples)), format_func=labels.__getitem__
    )
    row = samples.iloc[[selection]]
    info = st.columns(2)
    if "Amount" in row:
        info[0].metric("Transaction Amount", f"{row['Amount'].iloc[0]:,.2f}")
    if "Time" in row:
        info[1].metric("Elapsed Time (seconds)", f"{row['Time'].iloc[0]:,.0f}")

    if st.button("Analyze Transaction", type="primary"):
        try:
            scores, predictions = predict_transactions(
                model, threshold, feature_names, fraud_column, row
            )
            st.session_state["single_result"] = (
                selection, float(scores[0]), int(predictions[0])
            )
        except Exception as exc:
            st.session_state.pop("single_result", None)
            st.error(f"Could not analyze this transaction: {exc}")

    result = st.session_state.get("single_result")
    if result is not None and result[0] == selection:
        _, score, prediction = result
        actual = int(row["ActualClass"].iloc[0])
        classification = "FRAUD" if prediction else "NORMAL"
        metrics = st.columns(2)
        metrics[0].metric("Model Fraud Score", f"{score:.4f}")
        metrics[1].metric("Decision Threshold", f"{threshold:g}")
        st.markdown(f"**Model Classification:** {classification}")
        st.markdown(f"**Actual Classification:** {'FRAUD' if actual else 'NORMAL'}")
        if prediction == actual:
            st.success("Correct prediction")
        elif prediction:
            st.warning("Incorrect prediction — False Positive: a normal transaction was flagged as fraud.")
        else:
            st.warning("Incorrect prediction — Missed Fraud: a fraudulent transaction was classified as normal.")
        operator = ">=" if prediction else "<"
        st.code(f"{score:.4f} {operator} {threshold:g}\nClassification: {classification}", language="text")
        st.caption(SCORE_NOTE)
    else:
        st.caption("Select an example and analyze it to see a fresh model score and classification.")

    with st.expander("View Model Features"):
        inputs = model_inputs(row, feature_names)
        st.dataframe(
            pd.DataFrame({"Feature": feature_names, "Value": inputs.iloc[0].to_numpy()}),
            hide_index=True,
        )


def batch_prediction_tab(model, threshold, feature_names, fraud_column):
    st.write("Upload a CSV containing the 30 model features. Labels are optional; extra columns are excluded from inference.")
    st.caption("Required features, in model order: " + ", ".join(feature_names))
    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded is None:
        return
    try:
        frame = read_transactions(uploaded.getvalue())
        scores, predictions = predict_transactions(
            model, threshold, feature_names, fraud_column, frame
        )
    except Exception as exc:
        st.error(f"Could not analyze the uploaded CSV: {exc}")
        return

    output = frame.copy()
    # Preserve uploaded values even if the input already has result columns.
    for name in ["FraudScore", "Prediction"]:
        if name in output:
            replacement = "Uploaded" + name
            while replacement in output:
                replacement = "Uploaded" + replacement
            output = output.rename(columns={name: replacement})
            st.caption(f"Uploaded {name} retained as {replacement} in the results.")
    output["FraudScore"] = scores
    output["Prediction"] = np.where(predictions == 1, "FRAUD", "NORMAL")
    metrics = st.columns(3)
    metrics[0].metric("Transactions analyzed", len(output))
    metrics[1].metric("Transactions flagged as fraud", int(predictions.sum()))
    metrics[2].metric("Transactions classified as normal", int((predictions == 0).sum()))
    st.caption(f"Decision threshold: {threshold:g}. " + SCORE_NOTE)
    st.dataframe(output, hide_index=True)
    st.download_button(
        "Download Prediction Results",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="fraud_prediction_results.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(page_title="Credit Card Fraud Detection Demo", layout="centered")
    st.title("Credit Card Fraud Detection Demo")
    st.write("This demo uses the Balanced Random Forest model trained in this project to classify example credit card transactions.")
    st.caption("V1–V28 are anonymized PCA-transformed features. This is an ML project demonstration, not a real banking fraud detection service.")
    try:
        model, threshold, feature_names, fraud_column = load_model_package(MODEL_PATH)
    except Exception as exc:
        st.error(f"Could not load the saved model from models/credit_card_fraud_model.joblib: {exc}")
        st.stop()

    with st.sidebar:
        st.subheader("Model Information")
        st.write("**Model:** Balanced Random Forest")
        st.metric("Decision Threshold", f"{threshold:g}")
        st.subheader("Final Test Set Results")
        st.write("Precision: 0.8615\n\nRecall: 0.7887\n\nF1: 0.8235\n\nAverage Precision: 0.7861")
        st.caption("Recorded held-out test results; performance on new data may differ.")

    single, batch = st.tabs(["Single Transaction", "Batch Prediction"])
    with single:
        single_transaction_tab(model, threshold, feature_names, fraud_column)
    with batch:
        batch_prediction_tab(model, threshold, feature_names, fraud_column)


if __name__ == "__main__":
    main()
