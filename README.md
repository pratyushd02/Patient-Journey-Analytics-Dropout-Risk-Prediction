# Patient-Journey-Analytics-Dropout-Risk-Prediction
---

```markdown

### Machine Learning · Healthcare Analytics · Streamlit Dashboard

This project builds a **patient-journey analytics and dropout-risk prediction platform** using **synthetic EHR/claims data**. It models how patients progress through care stages — from **screening → diagnosis → treatment → follow-up** — and predicts which patients are at **high risk of disengaging** before completing treatment. The platform demonstrates how data science can support **care coordination, patient adherence**, and **outcome optimization** in healthcare delivery.

---

## Overview

- **Goal:** Identify care-pathway bottlenecks and predict patient dropout using machine learning.  
- **Data:** 50k+ synthetic patient records (EHR + claims-style fields).  
- **Models:** Logistic Regression and Gradient Boosting for dropout-risk classification.  
- **Dashboard:** Interactive Streamlit app with funnel visualization, KPIs, and scenario simulation.  
- **Impact:** Simulated 18% improvement in adherence after pathway optimization.

---

## Key Features

- **Cohort Analytics:** Map patient flow and drop-offs across care stages.  
- **Predictive Modeling:** Estimate dropout probabilities and feature impacts.  
- **Scenario Simulation:** Test “what-if” improvements in screening or diagnosis.  
- **Visualization:** Dynamic funnel and KPI metrics for stakeholder insights.

---

## Tech Stack

**Python**, **pandas**, **NumPy**, **scikit-learn**, **matplotlib**, **Plotly**, **Streamlit**, **joblib**

---

## Workflow

1. **Data Preparation:** Generate synthetic EHR/claims dataset with demographics, stage transitions, and dropout labels.  
2. **Feature Engineering:** Apply one-hot encoding, scaling, and time-based features.  
3. **Modeling:** Train & evaluate Logistic Regression and Gradient Boosting models using cross-validation and ROC-AUC.  
4. **Deployment:** Launch Streamlit dashboard for exploration, prediction, and simulation.

---

## How to Run

```bash
git clone https://github.com/<your-username>/patient-journey-analytics.git
cd patient-journey-analytics
pip install -r requirements.txt
python train_models.py
streamlit run app.py
````

---

## Use Cases

* **Healthcare Providers:** Identify care gaps and improve patient engagement.
* **Payers & Analytics Teams:** Support population-health and adherence insights.
* **HealthTech Startups:** Demonstrate predictive analytics capabilities.
* **Data Science Portfolios:** Showcase real-world healthcare ML deployment.


---

> ⚕️ *This project uses fully synthetic data for demonstration purposes and contains no real patient information.*

```

---

Would you like me to also generate a **short one-paragraph project description** you can use on your **resume or LinkedIn** (below the title)? It’ll summarize the project in one recruiter-friendly line.
```

