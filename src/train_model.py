from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

DATA_PATH = Path("data/student-mat.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Keep the first version simple and use features available before final exams.
FEATURES = [
    "age", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "higher", "internet", "goout",
    "health", "absences"
]

df = pd.read_csv(DATA_PATH, sep=";")

# UCI final grade G3 ranges from 0 to 20. We define below 10 as at risk.
# G1, G2 and G3 are deliberately NOT model inputs to avoid target leakage.
df["at_risk"] = (df["G3"] < 10).astype(int)

X = df[FEATURES].copy()
y = df["at_risk"]

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

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

metrics = {
    "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
    "precision_at_risk": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
    "recall_at_risk": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
    "f1_at_risk": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "features_used": FEATURES,
    "positive_class": "At risk (G3 < 10)"
}

print("\nMODEL PERFORMANCE")
for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0
))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, MODEL_DIR / "student_risk_model.joblib")
(OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=4))

print("\nSaved model to models/student_risk_model.joblib")
print("Saved metrics to outputs/metrics.json")
