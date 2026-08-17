"""
train_models.py
----------------
Trains 5 classification models on the "Predict Students' Dropout and Academic
Success" dataset (UCI ML Repo, id=697), evaluates them with 6 metrics, and
saves:
  - model/*.pkl          -> trained model pipelines (preprocessing + estimator)
  - model/label_encoder.pkl
  - model/feature_columns.json
  - model/metrics_summary.csv
  - data/test_data.csv    -> held-out test split (features + true Target), used by the Streamlit app

Run:
    python model/train_models.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(ROOT, "dataset.csv")
MODEL_DIR = HERE
DATA_DIR = ROOT

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    X = df.drop(columns=["Target"])
    y = df["Target"]
    return X, y


def build_models():
    """Return dict of {name: sklearn Pipeline(scaler + estimator)}."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)),
            ]
        ),
        "kNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15)),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", GaussianNB()),
            ]
        ),
        "Random Forest (Ensemble)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
                    ),
                ),
            ]
        ),
    }


def evaluate(model, X_test, y_test_enc, classes):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test_enc, y_pred)
    prec = precision_score(y_test_enc, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test_enc, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test_enc, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_test_enc, y_pred)
    try:
        auc = roc_auc_score(y_test_enc, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = np.nan

    return {
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4) if not np.isnan(auc) else None,
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
    }


def main():
    X, y = load_data()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    # Save held-out test data (features + true label as text) for the Streamlit app.
    test_df = X_test.copy()
    test_df["Target"] = le.inverse_transform(y_test)
    test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

    models = build_models()
    results = {}

    for name, pipe in models.items():
        print(f"Training {name} ...")
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test, le.classes_)
        results[name] = metrics
        print(f"  {metrics}")

        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipe, os.path.join(MODEL_DIR, f"{fname}.pkl"))

    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(list(X.columns), f, indent=2)

    summary = pd.DataFrame(results).T
    summary.index.name = "ML Model Name"
    summary.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"))
    print("\n=== Metrics Summary ===")
    print(summary)
    print("\nSaved models, encoder, feature list, and metrics summary to:", MODEL_DIR)
    print("Saved held-out test data to:", os.path.join(ROOT, "test_data.csv"))


if __name__ == "__main__":
    main()
