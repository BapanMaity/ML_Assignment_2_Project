# Student Dropout & Academic Success — Multi-Model Classifier

**BITS Pilani WILP · M.Tech (AIML/DSE) · Machine Learning · Assignment 2**

## a. Problem Statement

Higher-education institutions lose a significant number of students to
dropout every year, which hurts both the students' futures and the
institution's outcomes. Early identification of at-risk students allows
timely intervention. This project frames that problem as a **3-class
classification task**: given a student's demographic, socio-economic, and
academic-performance data collected at enrollment and after the first two
semesters, predict whether the student will ultimately be a **Dropout**,
remain **Enrolled**, or **Graduate**.

Five classical ML classifiers are trained on the same dataset, evaluated
with a consistent set of metrics, and made explorable through an
interactive Streamlit app.

## b. Dataset Description

- **Source:** UCI Machine Learning Repository — [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (Realinho et al., 2021, CC BY 4.0)
- **Instances:** 4,424 students enrolled between 2008–2019 across 17 undergraduate programs (agronomy, design, education, nursing, journalism, management, social service, technologies, etc.)
- **Features:** 36 input features spanning 4 groups:
  - **Demographic:** marital status, nationality, gender, age at enrollment, displaced/international status
  - **Socio-economic:** parents' qualification/occupation, scholarship holder, debtor status, tuition-fee status
  - **Macroeconomic:** unemployment rate, inflation rate, GDP at time of enrollment
  - **Academic:** admission grade, previous qualification, curricular units enrolled/approved/evaluated/grade for 1st and 2nd semesters
- **Target classes:** `Dropout` (1,421), `Enrolled` (794), `Graduate` (2,209) — a moderately imbalanced 3-class problem
- **Missing values:** None
- **Train/test split:** 80% / 20%, stratified by target class, `random_state=42`

## c. GitHub Repository Link

> **[REPLACE WITH YOUR ACTUAL GITHUB REPO LINK AFTER PUSHING]**
> `https://github.com/<your-username>/<your-repo-name>`

Repository contains: complete source code, `requirements.txt`, this
`README.md`, `data/test_data.csv` (held-out test split), and the
`model/` folder with training code and saved model files.

## d. Models Used

All 5 models are trained on an **identical 80/20 stratified split** of the
same dataset, with features standardized via `StandardScaler` inside a
`sklearn.Pipeline` (fit only on the training fold).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7684 | 0.8778 | 0.7070 | 0.6754 | 0.6826 | 0.6150 |
| Decision Tree | 0.7277 | 0.8115 | 0.6724 | 0.6572 | 0.6623 | 0.5527 |
| kNN | 0.7028 | 0.8135 | 0.6472 | 0.5889 | 0.5955 | 0.5019 |
| Naive Bayes | 0.6588 | 0.7893 | 0.5675 | 0.5579 | 0.5576 | 0.4279 |
| Random Forest (Ensemble) | 0.7751 | 0.8902 | 0.7359 | 0.6836 | 0.6967 | 0.6269 |

*(Precision, Recall, F1 are macro-averaged across the 3 classes; AUC is
macro-averaged one-vs-rest.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, well-balanced baseline (2nd highest AUC). The 3 classes are largely linearly separable given the engineered academic features (esp. 2nd-semester approved units and grades), so a linear decision boundary already captures most of the signal. |
| Decision Tree | Reasonable accuracy but visibly overfits relative to Random Forest — a single tree (even depth-limited) has higher variance and lower MCC than the ensembled version of itself. |
| kNN | Middling performance; sensitive to the mixed categorical/numeric feature scaling and the curse of dimensionality with 36 features. Distance-based similarity is a weaker signal here than the students' actual academic trajectory. |
| Naive Bayes | Weakest performer. The features (e.g., 1st- and 2nd-semester curricular unit counts, grades) are highly correlated with each other, which directly violates Gaussian NB's conditional-independence assumption and depresses its scores. |
| Random Forest (Ensemble) | **Best performer across every metric.** Averaging many decorrelated trees reduces the variance problem seen in the single Decision Tree while still capturing non-linear interactions between academic and socio-economic features. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — highest Accuracy (0.775), AUC (0.890), and MCC (0.627), and the most robust choice given the moderate class imbalance (MCC is a more reliable summary than accuracy here). |

## Repository Structure

```
project-folder/
│-- app.py                     # Streamlit app entry point
│-- requirements.txt
│-- README.md
│-- dataset.csv                # full cleaned dataset (4,424 rows, 36 features)
│-- test_data.csv              # held-out 20% test split — upload this in the app
│-- model/
│   │-- train_models.py        # trains all 5 models + saves metrics
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
│   │-- label_encoder.pkl
│   │-- feature_columns.json
│   │-- metrics_summary.csv
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # optional: retrain models from scratch
streamlit run app.py
```

## How to Use the Streamlit App

1. Select a model from the sidebar dropdown (Logistic Regression, Decision
   Tree, kNN, Naive Bayes, or Random Forest).
2. Upload `test_data.csv` (or any CSV with the same feature columns
   plus a `Target` column) using the sidebar file uploader.
3. View predictions, live evaluation metrics (Accuracy, AUC, Precision,
   Recall, F1, MCC), the confusion matrix, and the full per-class
   classification report.

## Deployment

Deployed on **Streamlit Community Cloud**:
`https://streamlit.io/cloud` → connect GitHub → select this repo → set
main file to `app.py` → Deploy.

> **Live app link:** `https://zqsqjsy2exdattpq5z47yd.streamlit.app/`

## Dataset Citation

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021).
*Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C5MC89
