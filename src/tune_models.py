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
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, balanced_accuracy_score
)

DATA_PATH = Path("data/student-mat.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Keep these 12 inputs aligned with the existing Streamlit form.
# G1, G2 and G3 are excluded from predictors to prevent target leakage.
FEATURES = [
    "age", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "higher", "internet", "goout",
    "health", "absences"
]

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Run 'python download_data.py' from the project root."
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

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "precision_at_risk": "precision",
    "recall_at_risk": "recall",
    "f1_at_risk": "f1",
    "balanced_accuracy": "balanced_accuracy"
}

search_spaces = {
    "Logistic Regression": (
        LogisticRegression(max_iter=3000, solver="liblinear"),
        {
            "classifier__C": [0.01, 0.1, 1, 10, 50],
            "classifier__class_weight": [None, "balanced"]
        }
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=1),
        {
            "classifier__n_estimators": [300, 600],
            "classifier__max_depth": [None, 4, 8],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__class_weight": [None, "balanced"]
        }
    ),
    "Extra Trees": (
        ExtraTreesClassifier(random_state=42, n_jobs=1),
        {
            "classifier__n_estimators": [300, 600],
            "classifier__max_depth": [None, 4, 8],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__class_weight": [None, "balanced"]
        }
    ),
    "Support Vector Machine": (
        SVC(kernel="rbf", probability=True),
        {
            "classifier__C": [0.1, 1, 10, 50],
            "classifier__gamma": ["scale", 0.01, 0.1],
            "classifier__class_weight": [None, "balanced"]
        }
    )
}

best_overall = None
search_summaries = []
print("\nHYPERPARAMETER TUNING USING TRAINING DATA ONLY")
print("Selection metric: mean 5-fold cross-validation F1-score for the at-risk class.")
print("The existing 20% test split is not used during tuning.\n")

for model_name, (classifier, param_grid) in search_spaces.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1_at_risk",
        cv=cv,
        n_jobs=1,
        return_train_score=False,
        error_score="raise"
    )
    print(f"Tuning {model_name}...")
    search.fit(X_train, y_train)
    idx = search.best_index_
    row = {
        "model": model_name,
        "cv_accuracy_mean": float(search.cv_results_["mean_test_accuracy"][idx]),
        "cv_precision_at_risk_mean": float(search.cv_results_["mean_test_precision_at_risk"][idx]),
        "cv_recall_at_risk_mean": float(search.cv_results_["mean_test_recall_at_risk"][idx]),
        "cv_f1_at_risk_mean": float(search.cv_results_["mean_test_f1_at_risk"][idx]),
        "cv_balanced_accuracy_mean": float(search.cv_results_["mean_test_balanced_accuracy"][idx]),
        "best_params": search.best_params_
    }
    search_summaries.append(row)
    print(
        f"  CV accuracy={row['cv_accuracy_mean']:.3f} | "
        f"At-risk precision={row['cv_precision_at_risk_mean']:.3f} | "
        f"recall={row['cv_recall_at_risk_mean']:.3f} | "
        f"F1={row['cv_f1_at_risk_mean']:.3f}"
    )
    if best_overall is None or row["cv_f1_at_risk_mean"] > best_overall["row"]["cv_f1_at_risk_mean"]:
        best_overall = {"name": model_name, "row": row, "model": search.best_estimator_}

best_model = best_overall["model"]
predictions = best_model.predict(X_test)

metrics = {
    "selected_model": best_overall["name"],
    "selection_method": "Highest mean 5-fold CV F1-score for at-risk class, using training split only",
    "tuning_results": search_summaries,
    "test_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
    "test_balanced_accuracy": round(float(balanced_accuracy_score(y_test, predictions)), 4),
    "test_precision_at_risk": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
    "test_recall_at_risk": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
    "test_f1_at_risk": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "features_used": FEATURES,
    "positive_class": "At risk (G3 < 10)"
}

print("\nSELECTED MODEL:", best_overall["name"])
print("BEST PARAMETERS:", best_overall["row"]["best_params"])
print("\nHOLDOUT TEST PERFORMANCE")
print(f"Accuracy:             {metrics['test_accuracy']:.2%}")
print(f"Balanced accuracy:    {metrics['test_balanced_accuracy']:.2%}")
print(f"At-risk precision:    {metrics['test_precision_at_risk']:.2%}")
print(f"At-risk recall:       {metrics['test_recall_at_risk']:.2%}")
print(f"At-risk F1-score:     {metrics['test_f1_at_risk']:.2%}")
print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0
))
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))

joblib.dump(best_model, MODEL_DIR / "student_risk_model.joblib")
(OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=4))
print("\nSaved tuned model to models/student_risk_model.joblib")
print("Saved tuning results to outputs/metrics.json")
