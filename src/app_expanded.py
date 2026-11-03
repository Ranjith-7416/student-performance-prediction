import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_risk_model.joblib"
METRICS_PATH = ROOT / "outputs" / "feature_model_metrics.json"

FEATURES = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet", "romantic",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
]

st.set_page_config(
    page_title="Student Performance Risk Predictor",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Performance Risk Predictor")
st.write(
    "Enter a student's background and current circumstances to estimate "
    "whether the student may be at risk of receiving a final grade below 10."
)

st.warning(
    "Educational demonstration only. This model is trained on a small dataset "
    "from two Portuguese schools. It is not validated for real decisions, "
    "and must not be used to label, penalize, or deny support to students."
)

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Run "
        "`python src/feature_engineering_model.py` from the project folder first."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

with st.form("student_details"):
    st.subheader("1. Student information")
    c1, c2, c3 = st.columns(3)

    with c1:
        school = st.selectbox("School", ["GP", "MS"], format_func=lambda x: {
            "GP": "Gabriel Pereira (GP)", "MS": "Mousinho da Silveira (MS)"
        }[x])
        sex = st.selectbox("Sex", ["F", "M"], format_func=lambda x: {
            "F": "Female", "M": "Male"
        }[x])
        age = st.slider("Age", 15, 22, 17)
        address = st.selectbox("Home area", ["U", "R"], format_func=lambda x: {
            "U": "Urban", "R": "Rural"
        }[x])
        famsize = st.selectbox("Family size", ["GT3", "LE3"], format_func=lambda x: {
            "GT3": "More than 3", "LE3": "3 or fewer"
        }[x])

    with c2:
        pstatus = st.selectbox("Parents' cohabitation status", ["T", "A"], format_func=lambda x: {
            "T": "Living together", "A": "Apart"
        }[x])
        medu = st.select_slider("Mother's education level", options=[0, 1, 2, 3, 4], value=2)
        fedu = st.select_slider("Father's education level", options=[0, 1, 2, 3, 4], value=2)
        mjob = st.selectbox("Mother's job", ["teacher", "health", "services", "at_home", "other"])
        fjob = st.selectbox("Father's job", ["teacher", "health", "services", "at_home", "other"])

    with c3:
        reason = st.selectbox("Reason for choosing school", ["course", "home", "reputation", "other"])
        guardian = st.selectbox("Main guardian", ["mother", "father", "other"])
        traveltime = st.select_slider("Travel time to school", options=[1, 2, 3, 4], value=1)
        studytime = st.select_slider("Weekly study time", options=[1, 2, 3, 4], value=2)
        failures = st.select_slider("Past class failures", options=[0, 1, 2, 3], value=0)

    st.subheader("2. Support and activities")
    c1, c2, c3 = st.columns(3)

    with c1:
        schoolsup = st.selectbox("Extra educational support", ["yes", "no"])
        famsup = st.selectbox("Family educational support", ["yes", "no"])
        paid = st.selectbox("Extra paid classes", ["yes", "no"])
        activities = st.selectbox("Extracurricular activities", ["yes", "no"])

    with c2:
        nursery = st.selectbox("Attended nursery school", ["yes", "no"])
        higher = st.selectbox("Wants higher education", ["yes", "no"])
        internet = st.selectbox("Internet access at home", ["yes", "no"])
        romantic = st.selectbox("In a romantic relationship", ["yes", "no"])

    with c3:
        famrel = st.slider("Family relationship quality", 1, 5, 4)
        freetime = st.slider("Free time after school", 1, 5, 3)
        goout = st.slider("Going out with friends", 1, 5, 3)
        absences = st.number_input("School absences", min_value=0, max_value=75, value=4)

    st.subheader("3. Lifestyle and health")
    c1, c2, c3 = st.columns(3)
    with c1:
        dalc = st.slider("Workday alcohol consumption indicator", 1, 5, 1)
    with c2:
        walc = st.slider("Weekend alcohol consumption indicator", 1, 5, 1)
    with c3:
        health = st.slider("Self-reported health (1–5)", 1, 5, 3)

    submitted = st.form_submit_button("Predict student risk", type="primary")

if submitted:
    row = {
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": pstatus,
        "Medu": medu,
        "Fedu": fedu,
        "Mjob": mjob,
        "Fjob": fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": dalc,
        "Walc": walc,
        "health": health,
        "absences": absences,
    }

    input_df = pd.DataFrame([row], columns=FEATURES)

    try:
        prediction = int(model.predict(input_df)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_df)[0][1])

        st.subheader("Prediction")
        if prediction == 1:
            st.error("The model estimates that this student may be at risk.")
        else:
            st.success("The model estimates that this student may not be at risk.")

        if probability is not None:
            st.metric("Estimated at-risk probability", f"{probability:.1%}")
            st.caption(
                "This is the model's estimated probability, not a guarantee "
                "or a diagnosis of the student's academic outcome."
            )

        st.info(
            "Use this result only as a prompt for supportive human review. "
            "Discuss the student's circumstances and offer help where appropriate."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

with st.expander("About this model"):
    st.write(
        "The model predicts the binary target at_risk = 1 when the final grade "
        "(G3) is below 10, and 0 otherwise. G1, G2 and G3 are not used as "
        "prediction inputs. The training dataset contains 395 records from "
        "two Portuguese schools."
    )
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as file:
                metrics = json.load(file)
            st.write(f"Selected model: {metrics.get('selected_model', 'Not recorded')}")
            st.write(
                "Exploratory holdout accuracy: "
                f"{metrics.get('holdout_test_results', {}).get('accuracy', float('nan')):.1%}"
            )
        except (OSError, ValueError, TypeError):
            st.caption("Saved metrics could not be displayed.")
