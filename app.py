
import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

st.set_page_config(layout="wide", page_title="Care Pathway Utilization & Outcome Dashboard")

st.title("Care Pathway Utilization & Outcome Dashboard (Prototype)")
st.markdown("This prototype uses synthetic patient-level data to demonstrate funnel metrics, risk prediction, and scenario simulation.")

DATA_PATH = "synthetic_patient_journeys.csv"
MODEL_PATH = "models/"
os.makedirs(MODEL_PATH, exist_ok=True)

@st.cache_data
def load_data(path=DATA_PATH):
    return pd.read_csv(path)

df = load_data(DATA_PATH)

# Sidebar filters
st.sidebar.header("Filters")
age_min, age_max = int(df['age'].min()), int(df['age'].max())
age_sel = st.sidebar.slider("Age range", age_min, age_max, (30, 80))
sex_sel = st.sidebar.multiselect("Sex", options=df['sex'].unique(), default=list(df['sex'].unique()))

filtered = df[(df['age']>=age_sel[0]) & (df['age']<=age_sel[1]) & (df['sex'].isin(sex_sel))]

# Funnel display
st.header("Funnel Metrics (Selected Population)")
total = len(filtered)
screened = int(filtered['screening_done'].sum())
diagnosed = int(filtered['diagnosed'].sum())
started = int(filtered['treatment_started'].sum())
followed = int(filtered['follow_up_done'].sum())
cols = st.columns(5)
cols[0].metric("Total patients", total)
cols[1].metric("Screened", screened)
cols[2].metric("Diagnosed", diagnosed)
cols[3].metric("Treatment Started", started)
cols[4].metric("Follow-up", followed)

# Show top-level dataframe
st.subheader("Sample patient journeys (filtered)")
st.dataframe(filtered.head(50))

# Risk prediction (pre-trained models saved with the project)
st.header("Dropout Risk Prediction (Diagnosed patients)")
st.markdown("Predict probability a diagnosed patient will *not* start treatment (dropout) based on age, comorbidity score, area index and time to diagnosis.")

# Load pre-trained model artifacts if present
try:
    with open(os.path.join(MODEL_PATH, "scaler.pkl"), "rb") as f:
        scaler = joblib.load(f)
    with open(os.path.join(MODEL_PATH, "logreg.pkl"), "rb") as f:
        logreg = joblib.load(f)
    with open(os.path.join(MODEL_PATH, "gb.pkl"), "rb") as f:
        gb = joblib.load(f)
    model_ready = True
except Exception as e:
    st.warning("Model artifacts not found. Please train models using the provided training script or load artifacts into the 'models' folder.")
    model_ready = False

# Allow manual input for a single patient
st.subheader("Patient risk calculator")
age_i = st.number_input("Age", min_value=18, max_value=100, value=60)
sex_i = st.selectbox("Sex", options=["F","M"])
comorb_i = st.number_input("Comorbidity score (0-8)", min_value=0, max_value=8, value=1)
adi_i = st.number_input("Area Deprivation Index", min_value=0.0, max_value=100.0, value=50.0)
ttd_i = st.number_input("Time to diagnosis (days)", min_value=0, max_value=365, value=10)

if st.button("Compute risk"):
    sample = pd.DataFrame([{
        "age": age_i,
        "comorbidity_score": comorb_i,
        "area_deprivation_index": adi_i,
        "time_to_diagnosis": ttd_i,
        "sex_M": 1 if sex_i=="M" else 0
    }])
    if model_ready:
        # apply scaler to numeric columns
        num_cols = ['age','comorbidity_score','area_deprivation_index','time_to_diagnosis']
        sample[num_cols] = scaler.transform(sample[num_cols])
        p_lr = logreg.predict_proba(sample)[:,1][0]
        p_gb = gb.predict_proba(sample)[:,1][0]
        st.metric("Logistic Regression - dropout probability", f"{p_lr:.2%}")
        st.metric("Gradient Boosting - dropout probability", f"{p_gb:.2%}")
    else:
        st.info("Model artifacts not available in this prototype. Check README for training instructions.")

# Scenario simulation: increase screening by X%
st.header("What-if Scenario: Improve Screening Outreach")
inc = st.slider("Increase screening rate by (%)", 0, 50, 10)
if st.button("Simulate impact"):
    # naive simulation: assume increased screening increases screened count proportionally among eligible population
    baseline_screened = filtered['screening_done'].sum()
    new_screened = min(len(filtered), int(baseline_screened * (1 + inc/100.0)))
    st.write(f"Screened: {baseline_screened} → {new_screened} (increase {inc}%)")
    # rough downstream effect: assume diagnosis and treatment scale with screening proportionally
    baseline_diagnosed = filtered['diagnosed'].sum()
    new_diagnosed = int(baseline_diagnosed * (new_screened / max(1, baseline_screened)))
    st.write(f"Diagnosed (approx): {baseline_diagnosed} → {new_diagnosed}")

st.markdown("---")
st.write("Prototype created for demonstration. See /mnt/data/health_project for artifacts (CSV, models, summary).")
