from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

DATA_PATH = Path("data/student-mat.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Only information available before the final grade is known.
# G1, G2 and G3 are deliberately excluded to prevent target leakage.
FEATURES = [
    "age", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "higher", "internet", "goout",
    "health", "absences"
]

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Run 'python download_data.py' from the project root first."
    )

df = pd.read_csv(DATA_PATH, sep=";")
df["at_risk"] = (df["G3"] < 10).astype(int)
X = df[FEATURES].copy()
y = df["at_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessing is inside each pipeline, so every CV fold learns its own
# imputation, scaling and category encoding without seeing validation data.
def make_pipeline(classifier, scale_numeric=True):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

candidates = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(max_iter=2000, class_weight=None)
    ),
    "Balanced Logistic Regression": make_pipeline(
        LogisticRegression(max_iter=2000, class_weight="balanced")
    ),
    "Random Forest": make_pipeline(
        RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
        scale_numeric=False
    ),
    "Extra Trees": make_pipeline(
        ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        scale_numeric=False
    ),
    "Support Vector Machine": make_pipeline(
        SVC(C=1.0, kernel="rbf", class_weight=None)
    ),
    "Balanced Support Vector Machine": make_pipeline(
        SVC(C=1.0, kernel="rbf", class_weight="balanced")
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "precision_at_risk": "precision",
    "recall_at_risk": "recall",
    "f1_at_risk": "f1",
}

results = []
print("\n5-FOLD CROSS-VALIDATION ON TRAINING DATA")
print("(The 20% test set is kept separate until final evaluation.)\n")

for name, pipeline in candidates.items():
    scores = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=scoring,
        n_jobs=1, error_score="raise"
    )
    row = {
        "model": name,
        "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
        "cv_accuracy_std": float(scores["test_accuracy"].std()),
        "cv_precision_mean": float(scores["test_precision_at_risk"].mean()),
        "cv_recall_mean": float(scores["test_recall_at_risk"].mean()),
        "cv_f1_mean": float(scores["test_f1_at_risk"].mean()),
    }
    results.append(row)
    print(
        f"{name:32s} "
        f"Accuracy: {row['cv_accuracy_mean']:.3f} +/- {row['cv_accuracy_std']:.3f} | "
        f"At-risk F1: {row['cv_f1_mean']:.3f} | "
        f"Recall: {row['cv_recall_mean']:.3f}"
    )

# Select using training-only cross-validation accuracy, not test-set results.
best_row = max(results, key=lambda r: r["cv_accuracy_mean"])
best_name = best_row["model"]
best_model = candidates[best_name]
best_model.fit(X_train, y_train)
test_predictions = best_model.predict(X_test)

metrics = {
    "selected_model": best_name,
    "selection_method": "Highest mean 5-fold CV accuracy on training data",
    "cv_results": results,
    "test_accuracy": round(float(accuracy_score(y_test, test_predictions)), 4),
    "test_precision_at_risk": round(float(precision_score(y_test, test_predictions, zero_division=0)), 4),
    "test_recall_at_risk": round(float(recall_score(y_test, test_predictions, zero_division=0)), 4),
    "test_f1_at_risk": round(float(f1_score(y_test, test_predictions, zero_division=0)), 4),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "features_used": FEATURES,
    "positive_class": "At risk (G3 < 10)",
}

print("\nSELECTED MODEL:", best_name)
print("\nFINAL HOLDOUT TEST PERFORMANCE")
print(f"Accuracy:       {metrics['test_accuracy']:.2%}")
print(f"At-risk precision: {metrics['test_precision_at_risk']:.2%}")
print(f"At-risk recall:    {metrics['test_recall_at_risk']:.2%}")
print(f"At-risk F1-score:  {metrics['test_f1_at_risk']:.2%}")
print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test, test_predictions,
    target_names=["Not at risk", "At risk"], zero_division=0
))
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, test_predictions))

joblib.dump(best_model, MODEL_DIR / "student_risk_model.joblib")
(OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=4))
print("\nSaved selected model to models/student_risk_model.joblib")
print("Saved comparison and metrics to outputs/metrics.json")
