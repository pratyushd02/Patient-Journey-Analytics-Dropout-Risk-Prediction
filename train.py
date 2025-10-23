# train_models.py
import pandas as pd
import os, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------- CONFIG ----------
DATA_PATH = "synthetic_patient_journeys.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- LOAD DATA ----------
df = pd.read_csv(DATA_PATH)

# Use only patients who have been diagnosed
diagnosed_df = df[df['diagnosed'] == 1].copy()

# Target: 1 if dropped out before treatment, 0 if started treatment
diagnosed_df['dropped_before_treatment'] = (diagnosed_df['treatment_started'] == 0).astype(int)

# ---------- FEATURES ----------
features = ['age', 'sex', 'comorbidity_score', 'area_deprivation_index', 'time_to_diagnosis']
X = pd.get_dummies(diagnosed_df[features], columns=['sex'], drop_first=True)
y = diagnosed_df['dropped_before_treatment']

# Scale numeric columns
num_cols = ['age', 'comorbidity_score', 'area_deprivation_index', 'time_to_diagnosis']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------- TRAIN MODELS ----------
logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg.fit(X_train, y_train)

gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
gb.fit(X_train, y_train)

# ---------- EVALUATE ----------
def evaluate(model, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"{name} -> Accuracy: {acc:.4f}, AUC: {auc:.4f}")

evaluate(logreg, "Logistic Regression")
evaluate(gb, "Gradient Boosting")

# ---------- SAVE ARTIFACTS ----------
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(logreg, os.path.join(MODEL_DIR, "logreg.pkl"))
joblib.dump(gb, os.path.join(MODEL_DIR, "gb.pkl"))

print(f"✅ Models and scaler saved to: {MODEL_DIR}")
