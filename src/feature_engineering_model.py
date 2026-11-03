"""
Feature-rich student risk model comparison.

Run from the project root:
    python src/feature_engineering_model.py

Target:
    at_risk = 1 when G3 < 10, otherwise 0.

Important:
    G1, G2 and G3 are excluded from predictors. G3 defines the target,
    while G1/G2 are excluded to preserve the early-prediction objective.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student-mat.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "student_risk_model.joblib"
METRICS_PATH = OUTPUT_DIR / "feature_model_metrics.json"

RANDOM_STATE = 42
TARGET = "at_risk"
EXCLUDED_COLUMNS = ["G1", "G2", "G3"]


def build_pipeline(model, numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", model),
        ]
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run python download_data.py first."
        )

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH, sep=";")
    df[TARGET] = (df["G3"] < 10).astype(int)

    # Keep all available pre-final-grade information.
    # G1/G2/G3 are deliberately excluded to prevent leakage and preserve
    # the project's early-warning prediction objective.
    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS + [TARGET]
    ]

    X = df[feature_columns].copy()
    y = df[TARGET].copy()

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    print("\nFEATURE-ENGINEERING MODEL COMPARISON")
    print("------------------------------------")
    print(f"Dataset rows: {len(df)}")
    print(f"Predictor columns: {len(feature_columns)}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Excluded columns: {EXCLUDED_COLUMNS}")
    print("Target: at_risk = 1 when G3 < 10")
    print("\nFeatures used:")
    print(", ".join(feature_columns))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Balanced Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=0.5,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Balanced SVM": SVC(
            C=1.0,
            kernel="rbf",
            class_weight="balanced",
        ),
    }

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision_at_risk": "precision",
        "recall_at_risk": "recall",
        "f1_at_risk": "f1",
        "balanced_accuracy": "balanced_accuracy",
    }

    results = []
    fitted_models = {}

    print("\nREPEATED 5-FOLD CROSS-VALIDATION ON TRAINING DATA")
    print("(3 repeats; holdout test set is not used for model selection.)")

    for name, estimator in models.items():
        pipeline = build_pipeline(
            estimator, numeric_features, categorical_features
        )

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        row = {
            "model": name,
            "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
            "cv_accuracy_std": float(scores["test_accuracy"].std()),
            "cv_precision_at_risk": float(
                scores["test_precision_at_risk"].mean()
            ),
            "cv_recall_at_risk": float(scores["test_recall_at_risk"].mean()),
            "cv_f1_at_risk": float(scores["test_f1_at_risk"].mean()),
            "cv_balanced_accuracy": float(
                scores["test_balanced_accuracy"].mean()
            ),
        }
        results.append(row)
        fitted_models[name] = pipeline

        print(
            f"\n{name}\n"
            f"  CV accuracy: {row['cv_accuracy_mean']:.3f} "
            f"+/- {row['cv_accuracy_std']:.3f}\n"
            f"  At-risk precision: {row['cv_precision_at_risk']:.3f}\n"
            f"  At-risk recall: {row['cv_recall_at_risk']:.3f}\n"
            f"  At-risk F1: {row['cv_f1_at_risk']:.3f}\n"
            f"  Balanced accuracy: {row['cv_balanced_accuracy']:.3f}"
        )

    # Select by at-risk F1, then recall, then accuracy.
    best_row = max(
        results,
        key=lambda item: (
            item["cv_f1_at_risk"],
            item["cv_recall_at_risk"],
            item["cv_accuracy_mean"],
        ),
    )
    best_name = best_row["model"]
    best_model = fitted_models[best_name]

    print(f"\nSELECTED MODEL: {best_name}")
    print("Selection priority: CV at-risk F1, then recall, then accuracy")

    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    test_results = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "precision_at_risk": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall_at_risk": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_at_risk": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["Not at risk", "At risk"],
            output_dict=True,
            zero_division=0,
        ),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    print("\nHOLDOUT TEST RESULTS (EXPLORATORY)")
    print("---------------------------------")
    print(f"Accuracy:             {test_results['accuracy']:.2%}")
    print(f"Balanced accuracy:    {test_results['balanced_accuracy']:.2%}")
    print(f"At-risk precision:    {test_results['precision_at_risk']:.2%}")
    print(f"At-risk recall:       {test_results['recall_at_risk']:.2%}")
    print(f"At-risk F1-score:     {test_results['f1_at_risk']:.2%}")
    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Not at risk", "At risk"],
            zero_division=0,
        )
    )
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(best_model, MODEL_PATH)

    output = {
        "selected_model": best_name,
        "selection_metric": "Repeated 5-fold CV at-risk F1",
        "feature_columns": feature_columns,
        "excluded_columns": EXCLUDED_COLUMNS,
        "cross_validation_results": results,
        "holdout_test_results": test_results,
        "note": (
            "The holdout split was examined during earlier development. "
            "Treat this result as exploratory, not as an independent final "
            "generalization estimate. G1, G2 and G3 were excluded from inputs."
        ),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print(f"\nSaved selected model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(
        "\nIMPORTANT: This selected model uses the expanded feature set. "
        "Do not run the existing app.py until its input form is updated "
        "to provide these additional features."
    )


if __name__ == "__main__":
    main()
