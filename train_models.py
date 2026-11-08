"""
Train five classifiers on the German Credit (Statlog) dataset and save them.

Run once:  python train_models.py
Outputs :  model/<name>.joblib   (preprocessing + classifier in a single pipeline)
           model/metrics_summary.csv
           test_data.csv         (held-out 20 % split, raw features + target)
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 2026
ROOT = Path(__file__).parent
TARGET = "default"          # 1 = applicant defaulted (bad credit), 0 = repaid

# ----------------------------------------------------------------------------
# 1. Load data and recode the target (original file uses 1 = good, 2 = bad)
# ----------------------------------------------------------------------------
df = pd.read_csv(ROOT / "data" / "german_credit.csv")
df[TARGET] = (df[TARGET] == 2).astype(int)

X = df.drop(columns=TARGET)
y = df[TARGET]

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

# test file used by the web app: raw features + target column
pd.concat([X_test, y_test], axis=1).to_csv(ROOT / "test_data.csv", index=False)


# ----------------------------------------------------------------------------
# 2. Preprocessing: scale numeric columns, one-hot encode categorical columns
# ----------------------------------------------------------------------------
def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


# ----------------------------------------------------------------------------
# 3. Candidate models with small hyper-parameter grids (5-fold CV on train set)
# ----------------------------------------------------------------------------
candidates = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"clf__C": [0.01, 0.1, 1, 10]},
    ),
    "decision_tree": (
        DecisionTreeClassifier(random_state=SEED),
        {"clf__max_depth": [3, 4, 5, 7, 10], "clf__min_samples_leaf": [1, 5, 10]},
    ),
    "knn": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [3, 5, 7, 11, 15, 21], "clf__weights": ["uniform", "distance"]},
    ),
    "naive_bayes": (
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 1e-1]},
    ),
    "svm": (
        SVC(probability=True, random_state=SEED),
        {"clf__C": [0.5, 1, 5, 10], "clf__gamma": ["scale", 0.01, 0.05]},
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
(ROOT / "model").mkdir(exist_ok=True)

rows, best_params = [], {}
for name, (estimator, grid) in candidates.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", estimator)])
    search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    best_params[name] = search.best_params_

    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    rows.append({
        "model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    joblib.dump(best, ROOT / "model" / f"{name}.joblib")
    print(f"{name:20s} saved   best params: {search.best_params_}")

summary = pd.DataFrame(rows).round(4)
summary.to_csv(ROOT / "model" / "metrics_summary.csv", index=False)
(ROOT / "model" / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))
print(summary.to_string(index=False))
print("train size:", len(X_train), "test size:", len(X_test))
