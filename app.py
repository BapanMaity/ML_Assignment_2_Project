"""
Streamlit app: Student Dropout & Academic Success — Multi-Model Classifier Demo

Features:
  a. Upload a CSV of test data (features + true 'Target' column)
  b. Select which trained model to use from a dropdown
  c. View evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  d. View confusion matrix + full classification report
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Student Dropout Classifier Demo", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}


@st.cache_resource
def load_label_encoder():
    return joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))


@st.cache_resource
def load_model(model_name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[model_name]))


@st.cache_data
def load_feature_columns():
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        return json.load(f)


@st.cache_data
def load_metrics_summary():
    return pd.read_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"), index_col=0)


st.title("🎓 Student Dropout & Academic Success — Classifier Demo")
st.caption(
    "BITS Pilani WILP · M.Tech (AIML/DSE) · Machine Learning · Assignment 2  |  "
    "Dataset: UCI ML Repo #697 — Predict Students' Dropout and Academic Success"
)

with st.expander("ℹ️ About this app", expanded=False):
    st.markdown(
        """
        This app demonstrates **5 classification models** trained on the same
        dataset to predict whether a student will **Dropout**, remain
        **Enrolled**, or **Graduate**, based on demographic, socio-economic,
        and academic features.

        Upload the provided `test_data.csv` (or your own CSV with the same
        columns, including the true `Target` column) to see how each model
        performs.
        """
    )

# ---- Sidebar controls -------------------------------------------------
st.sidebar.header("⚙️ Controls")

model_name = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

uploaded_file = st.sidebar.file_uploader(
    "Upload test data CSV (features + 'Target' column)", type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Pre-computed metrics (held-out test split)")
st.sidebar.dataframe(load_metrics_summary(), use_container_width=True)

# ---- Main panel ---------------------------------------------------------
feature_cols = load_feature_columns()
le = load_label_encoder()

if uploaded_file is None:
    st.info(
        "👈 Upload a test CSV from the sidebar to run predictions and see "
        "live evaluation metrics. You can use the `data/test_data.csv` "
        "file included in this repo."
    )
    st.subheader("Comparison of all 5 models (from training-time evaluation)")
    summary = load_metrics_summary()
    st.dataframe(summary, use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    summary[["Accuracy", "AUC", "F1", "MCC"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig)

else:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        st.error(f"Could not read the uploaded CSV: {e}")
        st.stop()

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        st.error(
            f"Uploaded CSV is missing {len(missing_cols)} required feature "
            f"column(s), e.g.: {missing_cols[:5]}"
        )
        st.stop()

    has_target = "Target" in df.columns

    X = df[feature_cols]
    model = load_model(model_name)

    with st.spinner(f"Running {model_name} on {len(df)} rows..."):
        y_pred_enc = model.predict(X)
        y_proba = model.predict_proba(X)
        y_pred_labels = le.inverse_transform(y_pred_enc)

    st.subheader(f"🔮 Predictions — {model_name}")
    result_df = df.copy()
    result_df["Predicted"] = y_pred_labels
    st.dataframe(result_df.head(50), use_container_width=True)
    st.caption(f"Showing first 50 of {len(result_df)} rows.")

    if has_target:
        y_true_enc = le.transform(df["Target"])

        acc = accuracy_score(y_true_enc, y_pred_enc)
        prec = precision_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)
        rec = recall_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)
        f1 = f1_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)
        mcc = matthews_corrcoef(y_true_enc, y_pred_enc)
        try:
            auc = roc_auc_score(y_true_enc, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")

        st.subheader("📈 Evaluation metrics on uploaded data")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("AUC", f"{auc:.4f}" if not np.isnan(auc) else "N/A")
        c3.metric("Precision", f"{prec:.4f}")
        c4.metric("Recall", f"{rec:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC", f"{mcc:.4f}")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true_enc, y_pred_enc)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                ax=ax_cm,
            )
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

        with col_b:
            st.subheader("Classification Report")
            report = classification_report(
                y_true_enc, y_pred_enc, target_names=le.classes_, output_dict=True
            )
            st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
    else:
        st.warning(
            "No 'Target' column found in the uploaded CSV — showing "
            "predictions only. Upload a CSV with a 'Target' column to see "
            "evaluation metrics and the confusion matrix."
        )

st.markdown("---")
st.caption("Built with Streamlit · scikit-learn · Deployed on Streamlit Community Cloud")
