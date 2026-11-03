"""
Academic-progress model: predict G3 < 10 using G1, G2 plus student details.

This is a different, legitimate prediction task from the earlier early-warning
model. G1 and G2 must be available when making the prediction. G3 is the target
and is NEVER included as an input.

Run from project root:
    python src/train_with_grades.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "student-mat.csv"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "student_risk_model_with_grades.joblib"
METRICS_PATH = OUT_DIR / "metrics_with_grades.json"
TARGET = "at_risk"


def make_pipeline(estimator, numeric, categorical):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])
    return Pipeline([("preprocessing", prep), ("classifier", estimator)])


def main():
    if not DATA.exists():
        raise FileNotFoundError("data/student-mat.csv not found. Run python download_data.py.")

    MODEL_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA, sep=";")
    df[TARGET] = (df["G3"] < 10).astype(int)

    # G1 and G2 are known prior-period grades. G3 is only the target.
    features = [c for c in df.columns if c not in ["G3", TARGET]]
    X = df[features].copy()
    y = df[TARGET].copy()

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", C=1.0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "RBF SVM": SVC(C=1.0, kernel="rbf", class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05,
            max_depth=2, random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "balanced_accuracy": "balanced_accuracy",
    }

    print("PREDICTING FINAL-GRADE RISK USING G1 + G2")
    print("Target: at_risk = 1 when G3 < 10")
    print(f"Rows: {len(df)} | Features: {len(features)}")
    print("G1 and G2 are inputs; G3 is excluded from inputs.")
    print("\n5-FOLD CROSS-VALIDATION (TRAINING DATA ONLY)")

    results = []
    pipelines = {}
    for name, estimator in models.items():
        pipe = make_pipeline(estimator, numeric, categorical)
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        row = {
            "model": name,
            "cv_accuracy": float(scores["test_accuracy"].mean()),
            "cv_accuracy_std": float(scores["test_accuracy"].std()),
            "cv_f1_at_risk": float(scores["test_f1"].mean()),
            "cv_recall_at_risk": float(scores["test_recall"].mean()),
            "cv_precision_at_risk": float(scores["test_precision"].mean()),
            "cv_balanced_accuracy": float(scores["test_balanced_accuracy"].mean()),
        }
        results.append(row)
        pipelines[name] = pipe
        print(
            f"{name:22s} accuracy={row['cv_accuracy']:.3f} "
            f"F1={row['cv_f1_at_risk']:.3f} "
            f"recall={row['cv_recall_at_risk']:.3f}"
        )

    # Select on CV F1, then accuracy. Holdout is not used for selection.
    best = max(results, key=lambda r: (r["cv_f1_at_risk"], r["cv_accuracy"]))
    selected_name = best["model"]
    selected = pipelines[selected_name]
    selected.fit(X_train, y_train)
    pred = selected.predict(X_test)

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision_at_risk": float(precision_score(y_test, pred, zero_division=0)),
        "recall_at_risk": float(recall_score(y_test, pred, zero_division=0)),
        "f1_at_risk": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(
            y_test, pred, target_names=["Not at risk", "At risk"],
            output_dict=True, zero_division=0
        ),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    joblib.dump(selected, MODEL_PATH)
    report = {
        "task": "Predict G3 < 10 using G1, G2 and student background features",
        "selected_model": selected_name,
        "features": features,
        "target": "G3 < 10",
        "cross_validation": results,
        "holdout_test": test_metrics,
        "caution": (
            "G1 and G2 must be available at prediction time. This is not an "
            "early prediction before prior-period grades. The dataset is small "
            "and from two Portuguese schools. Treat the test score as exploratory "
            "because this split has been examined in earlier development."
        ),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSELECTED MODEL: {selected_name}")
    print("\nHOLDOUT TEST RESULTS (EXPLORATORY)")
    print(f"Accuracy:          {test_metrics['accuracy']:.2%}")
    print(f"Balanced accuracy: {test_metrics['balanced_accuracy']:.2%}")
    print(f"At-risk precision: {test_metrics['precision_at_risk']:.2%}")
    print(f"At-risk recall:    {test_metrics['recall_at_risk']:.2%}")
    print(f"At-risk F1:        {test_metrics['f1_at_risk']:.2%}")
    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test, pred, target_names=["Not at risk", "At risk"], zero_division=0
    ))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, pred))
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")
    print("\nThis model requires G1 and G2 inputs. Do not use the old app form with it.")


if __name__ == "__main__":
    main()
