import streamlit as st

st.set_page_config(
    page_title="Triage support system",
    page_icon="",
)

st.title("Triage support system")

st.sidebar.success("Navigate the app above.")

st.markdown("""
## Business Problem & Goal\n
Singaporeans are famously **kiasu**, they have a cultural tendency to rush to Emergency Department "just in case".
**The result?** A system under serious strain, with genuinely critical patients losing out.
""")

col1, col2, col3 = st.columns(3)
col1.metric(label="ED visits per year", value="~1M")
col2.metric(label="Non-urgent cases", value="~40%")
col3.metric(label="Average wait time", value="4+ hours")

st.markdown("""
---
## Objectives
- **Pre-hospital Triage Utitlity** - Can patient-reported data provide meaningful pre-hospital triage guidance? Evaluates the predictive utility of 
limited features available to patients for self-assessment of symptom severity.
- **Clinical data Information Gain** - How much information is achieved using richer hospital data? Quantifies the diagnostic value added by 
clinical vitals, medication, arrival information, demographics recorded during professional triage.
- **Predictive Model Comparison** - Which machine learning model best predicts patient acuity? Systematic comparison of Linear Regression, 
Random Forest, XGBoost and MLP architecture for Emergency Severity Index (ESI) classification.
- **Safety & Interpretability Validation** - Are the models safe, reliable, and clinically interpretable? Validates model integrity via safety-critical recall, 
under triage rates, probabilistic calibration and SHAP based explainability.
            
---
            
## Problem Formulation
- **Learning Task**: Ordinal Classification problem. Labels: ESI-1 (Most urgent) -> ESI-5 (Least urgent).
- **Input (X)**: Chief complaint text, vital signs, pain score, demographics, arrival information, engineered features (More on Data page).
- **Output (Y)**: ESI-1 to ESI-5
            
---

## Explore this app
- **Inference** - Try the winning hospital or patient facing model, with the option to activate tuned thresholds.
- **Kaki Care** - Try the chat interface for users needing a quick acuity assessment with features collectible from your home.
- **Data** - Explore the data being used to train the models, features that were engineered and the two feature sets we used.
- **Model Comparisons** - See the performance of all 4 models we experimented, choose to see tuned / untuned versions.
""")