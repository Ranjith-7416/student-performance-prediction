# Student Performance Prediction Using Python

A beginner-friendly Data Science portfolio project. It explores student data and trains a machine-learning model to estimate whether a student may be at risk of failing Mathematics.

## Project goal

Use student information such as study time, past failures, school support, absences and other background factors to estimate academic risk.

## Dataset

This project uses the Mathematics (`student-mat.csv`) data from the **UCI Student Performance** dataset.

- Official dataset: https://archive.ics.uci.edu/dataset/320/student-performance
- Original creator: Paulo Cortez
- Dataset DOI: https://doi.org/10.24432/C5TG7T
- License: CC BY 4.0
- The full UCI collection contains records from two Portuguese secondary schools. This project uses the Mathematics subset (395 rows).

Run `python download_data.py` to download the dataset into `data/`.

## Important target definition

The dataset's final grade, `G3`, ranges from 0 to 20. For this project:

- `at_risk = 1` when `G3 < 10`
- `at_risk = 0` when `G3 >= 10`

The model does **not** use `G1`, `G2`, or `G3` as input features. `G1` and `G2` are earlier grades and are strongly related to `G3`; excluding them makes this a more realistic early-risk exercise and avoids target leakage.

## Tools

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit and Joblib.

## Folder structure

```text
student-performance-prediction/
├── data/                  # downloaded dataset (not committed to Git)
├── models/                # trained model (not committed to Git)
├── outputs/               # charts and evaluation metrics
├── src/
│   ├── eda.py             # data inspection and charts
│   └── train_model.py     # preprocessing, training and evaluation
├── app.py                 # Streamlit prediction application
├── download_data.py       # downloads official UCI data
├── requirements.txt
└── README.md
```

## Setup (macOS / Windows / Linux)

### 1. Open the project folder

```bash
cd student-performance-prediction
```

### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
py -m venv .venv
.venv\Scripts\activate
```

### 3. Install packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the dataset

```bash
python download_data.py
```

### 5. Explore the dataset

```bash
python src/eda.py
```

Open the generated charts in the `outputs/` folder.

### 6. Train and evaluate the model

```bash
python src/train_model.py
```

The trained model is saved to `models/student_risk_model.joblib`; evaluation metrics are saved to `outputs/metrics.json`.

### 7. Run the web application

```bash
streamlit run app.py
```

A browser tab should open with the student prediction application.

## Evaluation

The training script reports:

- Accuracy: overall share of correct predictions.
- Precision for the at-risk class: among students flagged as at risk, how many were actually at risk in the test data.
- Recall for the at-risk class: among students actually at risk, how many the model identified.
- F1-score: balance between precision and recall.
- Confusion matrix: counts of correct and incorrect predictions.

Do not claim a particular accuracy before running the model. Results depend on the data split and model.

## Limitations and responsible use

- This is a learning project, not a validated educational assessment tool.
- The dataset comes from two schools in Portugal; it may not represent students in India or other education systems.
- The dataset is relatively small.
- The model may reflect bias in the historical data.
- A prediction is not a fact about a student's ability or future.
- Do not use this model for admissions, grading, discipline, or other high-stakes decisions.

## Suggested portfolio improvements

1. Compare Logistic Regression with Decision Tree and Random Forest.
2. Use cross-validation.
3. Add a feature-importance or model-explanation section.
4. Test the model on a separate, locally collected dataset only with proper consent and privacy safeguards.
5. Add screenshots of the dashboard and explain the results in your portfolio.

## Citation

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
