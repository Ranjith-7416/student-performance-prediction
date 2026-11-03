from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path("data/student-mat.csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")

print("\nFIRST FIVE ROWS")
print(df.head())
print("\nDATASET SHAPE:", df.shape)
print("\nCOLUMN TYPES")
print(df.dtypes)
print("\nMISSING VALUES")
print(df.isnull().sum())
print("\nDUPLICATE ROWS:", df.duplicated().sum())
print("\nNUMERICAL SUMMARY")
print(df.describe())

# The final grade (G3) is the outcome used to create our pass/fail label.
df["at_risk"] = (df["G3"] < 10).map({True: "At risk", False: "Not at risk"})

sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 5))
sns.histplot(df["G3"], bins=15, kde=True)
plt.title("Distribution of Final Mathematics Grades")
plt.xlabel("Final grade (0–20)")
plt.tight_layout()
plt.savefig(OUT_DIR / "final_grade_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="at_risk")
plt.title("Students by Academic Risk Group")
plt.xlabel("Group")
plt.ylabel("Number of students")
plt.tight_layout()
plt.savefig(OUT_DIR / "risk_group_counts.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="studytime", y="G3")
plt.title("Study Time vs Final Grade")
plt.xlabel("Weekly study time category (1 = lowest, 4 = highest)")
plt.ylabel("Final grade (0–20)")
plt.tight_layout()
plt.savefig(OUT_DIR / "studytime_vs_grade.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="at_risk", y="absences")
plt.title("Absences by Academic Risk Group")
plt.xlabel("Group")
plt.ylabel("Number of absences")
plt.tight_layout()
plt.savefig(OUT_DIR / "absences_by_risk.png", dpi=150)
plt.close()

print("\nCharts saved in the outputs/ folder.")
print("\nRisk-group counts:")
print(df["at_risk"].value_counts())
