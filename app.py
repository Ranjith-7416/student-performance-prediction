"""
Streamlit app for the academic-progress student risk model.
Requires the model created by: python src/train_with_grades.py
Run from project root: streamlit run app.py
"""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student-mat.csv"
MODEL_PATH = ROOT / "models" / "student_risk_model_with_grades.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics_with_grades.json"

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Risk Prediction")
st.write(
    "Enter student details and the first- and second-period grades (G1 and G2). "
    "The model estimates whether the final grade (G3) may be below 10."
)

missing = [p for p in (DATA_PATH, MODEL_PATH) if not p.exists()]
if missing:
    st.error("Required project file(s) are missing: " + ", ".join(str(p.relative_to(ROOT)) for p in missing))
    st.info("First run `python src/train_with_grades.py` from the project folder.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, sep=";")

model = load_model()
df = load_data()
feature_columns = [c for c in df.columns if c != "G3"]

st.subheader("Enter student information")
st.caption("Use the same coding/scale as the UCI Student Performance dataset. G1 and G2 are grades from 0 to 20.")

with st.form("student_details"):
    values = {}
    numeric_columns = df[feature_columns].select_dtypes(include="number").columns.tolist()
    categorical_columns = df[feature_columns].select_dtypes(exclude="number").columns.tolist()

    left, right = st.columns(2)
    with left:
        st.markdown("**Academic and personal details**")
        for col in numeric_columns:
            low = int(df[col].min())
            high = int(df[col].max())
            default = int(df[col].median())
            if col in ("G1", "G2"):
                st.markdown(f"**{col} – {'First' if col == 'G1' else 'Second'} period grade**")
            values[col] = st.number_input(
                col, min_value=low, max_value=high, value=default, step=1,
                key=f"num_{col}"
            )
    with right:
        st.markdown("**Category-based details**")
        for col in categorical_columns:
            options = sorted(df[col].dropna().astype(str).unique().tolist())
            values[col] = st.selectbox(col, options=options, key=f"cat_{col}")

    submitted = st.form_submit_button("Predict student risk", type="primary", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([{col: values[col] for col in feature_columns}], columns=feature_columns)
    prediction = int(model.predict(input_df)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(probabilities[classes.index(1)])

    st.subheader("Prediction")
    if prediction == 1:
        st.error("At risk: the model predicts a final grade below 10.")
    else:
        st.success("Not at risk: the model predicts a final grade of 10 or above.")

    if probability is not None:
        st.metric("Estimated at-risk probability", f"{probability:.1%}")
        st.caption("This is a model probability, not a guarantee or a validated individual-level assessment.")

st.divider()
st.subheader("Model evaluation")
if METRICS_PATH.exists():
    with open(METRICS_PATH, encoding="utf-8") as f:
        report = json.load(f)
    test = report.get("holdout_test", {})
    a, b, c, d = st.columns(4)
    a.metric("Test accuracy", f"{test.get('accuracy', 0):.1%}")
    b.metric("Balanced accuracy", f"{test.get('balanced_accuracy', 0):.1%}")
    c.metric("At-risk recall", f"{test.get('recall_at_risk', 0):.1%}")
    d.metric("At-risk F1-score", f"{test.get('f1_at_risk', 0):.1%}")
    st.write(f"Selected model: **{report.get('selected_model', 'Not recorded')}**")
else:
    st.warning("Metrics file not found. Run model training first.")

st.warning(
    "Educational demonstration only. The dataset contains 395 records from two Portuguese schools. "
    "It is not validated for real-world student decisions or for students from other education systems."
)
