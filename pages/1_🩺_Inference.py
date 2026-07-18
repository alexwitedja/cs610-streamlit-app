import streamlit as st

from src.inference_pipeline import single_inference, cascade
from datetime import date, time, datetime
import pandas as pd

st.set_page_config(
    page_title="Inference",
    page_icon="🩺",
)

st.title("Try it")
st.markdown("Fill in the form or use the synthetic default values, then see what KiasuCare predicts.")

st.caption("Predictions are illustrative. This is a coursework model on US data — do not use it for real triage.")

st.sidebar.header("Inference")
mode = st.sidebar.radio(
    "Which model?",
    ["Patient-facing", "Hospital-facing"],
    help=(
        "Patient-facing — what someone could self-report from home. Fewer vitals, no clinical context.\n\n"
        "Hospital-facing — what the ED knows at the bedside. Full vitals and arrival context."
    ),
)

MED_LABELS = {
    "Diabetes": "diabetes", "Cardiac": "cardiac", "Respiratory": "respiratory",
    "Mental health, sleep, or anxiety": "psych", "Opioid": "opioid",
    "Anticonvulsant": "anticonvulsant", "Bloodthinner": "bloodthinner",
    "Thyroid": "thyroid", "Digestive": "gi",
}

pred_result = None
if mode == "Patient-facing":
    with st.form(key="patient_facing"):
        st.subheader("Patient-facing inference", help="Lean and limited number of features.")
        left, mid, right = st.columns(3)
        temperature = left.number_input("Temperature (°C)", 35.0, 42.0, 37.2, 0.1)
        heart_rate = mid.number_input("Heart rate (bpm)", 20, 300, 88)
        age = right.number_input("Age", 18, 103, 54)
        gender = left.selectbox("Gender", options=["Male", "Female"], index=1)
        pain_score = mid.number_input("Pain score", 0, 10, 6, help="0-10")
        right.markdown("<div style='height: 1.9rem'></div>", unsafe_allow_html=True)
        pain_missing = right.checkbox("Pain missing", help="Pain not reported")
        medication_history = st.multiselect(
            "Medication History", options=list(MED_LABELS), default=["Cardiac"],
            help="Based on the medicine you have taken.",
        )
        chief_complaint = st.text_area(
            "Chief complaint", value="abdominal pain and nausea since yesterday"
        )
        submitted = st.form_submit_button("Run", type="primary")

    if submitted:
        presentation = {
            "chief_complaint": chief_complaint,
            "age": age,
            "gender": "M" if gender == "Male" else "F",
            "temperature": temperature,
            "heartrate": heart_rate,
            "pain": None if pain_missing else pain_score,
            "medications": [MED_LABELS[m] for m in medication_history],
        }
        pred_result = single_inference(presentation, "patient")

else:
    with st.form(key="hospital_facing"):
        st.subheader("Hospital-facing inference", help="A more complete set of features.")
        left, mid, right = st.columns(3)
        temperature = left.number_input("Temperature (°C)", 35.0, 42.0, 37.2, 0.1)
        heart_rate = mid.number_input("Heart rate (bpm)", 20, 300, 88)
        age = right.number_input("Age", 18, 103, 54)
        gender = left.selectbox("Gender", options=["Male", "Female"], index=1)
        pain_score = mid.number_input("Pain score", 0, 10, 6, help="0-10")
        transport = right.selectbox(
            "Arrival Transport",
            options=["Ambulance", "Walk In", "Helicopter", "Other", "Unknown"], index=1,
        )
        resprate = left.number_input("Respiratory rate (breaths/min)", 5, 60, 18)
        o2sat = mid.number_input("SpO₂ (%)", 50, 100, 98)
        sbp = right.number_input("Systolic BP (mmHg)", 50, 300, 128)
        dbp = left.number_input("Diastolic BP (mmHg)", 20, 200, 76)
        arrival_date = mid.date_input("Arrival date", value=date.today())
        arrival_clock = right.time_input("Arrival time", value=time(14, 0))

        prior_ed_visit_all_time = left.number_input("Prior ED Visit (all time)", 0, 169, 1)
        prior_ed_visit_1_year = mid.number_input("Prior ED Visit (past 365d)", 0, 71, 0)
        prior_admissions_1_year = right.number_input("Prior Admissions (past 365d)", 0, 38, 0)
        last_ed_visit = left.date_input("Last ED Visit", value=date.today())

        med_count_cardiac = mid.number_input("Cardiac Medication Count", 0, 20, 1)
        med_count_diabetes = right.number_input("Diabetes Medication Count", 0, 20, 0)
        med_count_respiratory = left.number_input("Respiratory Medication Count", 0, 20, 0)
        med_count_psych = mid.number_input(
            "Psychology Medication Count", 0, 20, 0, help="Mental Health, Sleep, Anxiety"
        )
        med_count_opioid = right.number_input("Opioid Medication Count", 0, 20, 0)
        med_count_gi = left.number_input(
            "Digestive Medication Count", 0, 20, 0, help="Stomach Acid, Heartburn/Reflux, Nausea"
        )
        med_count_thyroid = mid.number_input("Thyroid Medication Count", 0, 20, 0)
        med_count_anticonvulsant = right.number_input("Anticonvulsant Medication Count", 0, 20, 0)
        med_count_bloodthinner = left.number_input("Bloodthinner Medication Count", 0, 20, 0)

        mid.markdown("<div style='height: 1.9rem'></div>", unsafe_allow_html=True)
        first_ed_visit = mid.checkbox("First ED visit?", value=True)
        pain_missing = right.checkbox("Pain missing", help="Pain not reported")
        last_visit_admission = right.checkbox("Last visit ended in admission")
        chief_complaint = st.text_area(
            "Chief complaint", value="abdominal pain and nausea since yesterday"
        )
        submitted = st.form_submit_button("Run", type="primary")

    if submitted:
        arrival = datetime.combine(arrival_date, arrival_clock)
        med_counts = {
            "cardiac": med_count_cardiac, "diabetes": med_count_diabetes,
            "respiratory": med_count_respiratory, "psych": med_count_psych,
            "opioid": med_count_opioid, "gi": med_count_gi,
            "thyroid": med_count_thyroid, "anticonvulsant": med_count_anticonvulsant,
            "bloodthinner": med_count_bloodthinner,
        }
        presentation = {
            "chief_complaint": chief_complaint,
            "age": age,
            "gender": "M" if gender == "Male" else "F",
            "temperature": temperature,
            "heartrate": heart_rate,
            "resprate": resprate,
            "o2sat": o2sat,
            "sbp": sbp,
            "dbp": dbp,
            "pain": None if pain_missing else pain_score,
            "med_counts": med_counts,
            "med_count_total": sum(med_counts.values()),
            "arrival_transport": transport.upper(),
            "arrival_hour": arrival.hour,
            "arrival_dow": arrival.weekday(),
            "arrival_month": arrival.month,
            "prior_ed_visits_total": 0 if first_ed_visit else prior_ed_visit_all_time,
            "prior_ed_visits_1yr": 0 if first_ed_visit else prior_ed_visit_1_year,
            "prior_admissions_1yr": 0 if first_ed_visit else prior_admissions_1_year,
            "days_since_last_ed": 9999 if first_ed_visit else (date.today() - last_ed_visit).days,
            "last_ed_admitted": int(last_visit_admission),
        }

        pred_result = single_inference(presentation, "hospital")

tuned = st.sidebar.checkbox(
    "Apply threshold tuning",
    help=(
        "Clinical safety threshold tuning\n\n"
        "Unchecked (raw) — the model's raw argmax, its single most likely class.\n\n"
        "Checked (tuned) — the priority cascade: if P(ESI-1) clears a tuned bar, escalate to ESI-1; "
        "otherwise if P(ESI-2) clears its bar, escalate to ESI-2; otherwise, argmax."
    ),
)

THRESHOLDS = {
    "Patient-facing": {"t1": "0.20", "t2": "0.25"},
    "Hospital-facing": {"t1": "0.50", "t2": "0.25"},
}
TUNING_COST = {
    "Patient-facing": {
        "recall_raw": "75.0%", "recall_tuned": "89.4%",
        "overtriage_raw": "15.2%", "overtriage_tuned": "31.0%",
        "qwk_raw": "0.63", "qwk_tuned": "0.57",
    },
    "Hospital-facing": {
        "recall_raw": "79.1%", "recall_tuned": "90.3%",
        "overtriage_raw": "17.5%", "overtriage_tuned": "27.6%",
        "qwk_raw": "0.62", "qwk_tuned": "0.59",
    },
}

t = THRESHOLDS[mode]
cost = TUNING_COST[mode]
side_label = "hospital" if mode == "Hospital-facing" else "patient"

if tuned:
    st.markdown(f"""
**The priority cascade is active.** If P(ESI-1) ≥ **{t['t1']}**, escalate to ESI-1. Otherwise if
P(ESI-2) ≥ **{t['t2']}**, escalate to ESI-2. Otherwise, argmax.

Tuning is not free. On the {side_label} model it lifts pooled critical recall from
**{cost['recall_raw']}** to **{cost['recall_tuned']}** — but over-triage rises from
**{cost['overtriage_raw']}** to **{cost['overtriage_tuned']}** and QWK falls from **{cost['qwk_raw']}**
to **{cost['qwk_tuned']}**. You are buying safety with clinician time.
""")
else:
    st.markdown("The model's raw argmax — its single most likely class.")

st.divider()

st.subheader("Result")

if pred_result:
    if mode == "Patient-facing":
        best_model = "MLP Ensemble"
    else:
        best_model = "Soft Voting Ensemble"

    if tuned:
        predicted_esi = cascade(pred_result[best_model])[0]
    else:
        predicted_esi = pred_result[best_model].argmax() + 1

    st.markdown(f"**Best model ({best_model}) predicted acuity: ESI {predicted_esi}**")
    st.text("Raw Probability Distribution across 4 models.")
    table = pd.DataFrame(
        {
            "Logistic Regression": pred_result["Logistic Regression"],
            "XGBoost": pred_result["XGBoost"],
            "MLP Ensemble": pred_result["MLP Ensemble"],
            "Soft Voting (MLP + XGB)": pred_result["Soft Voting Ensemble"],
        },
        index=["ESI-1", "ESI-2", "ESI-3", "ESI-4", "ESI-5"]
    )
    st.bar_chart(table)

st.caption(
    "The free text chief complaint is embedded with ClinicalBERT at inference time. The model is cached "
    "so it only loads once — loading it per-request would make the demo feel broken."
)

st.sidebar.markdown("### What this model was trained on")
if mode == "Patient-facing":
    st.sidebar.markdown("""
- Temperature, heart rate, pain score
- Demographics (age, gender, race)
- Medication history
- Chief complaint (ClinicalBERT embedding)
""")
else:
    st.sidebar.markdown("""
- Full core vitals + respiratory rate, SpO₂, blood pressure
- Demographics (age, gender, race)
- Medication history
- Arrival transport, ED visit history, 19 derived clinical features
- Chief complaint (ClinicalBERT embedding)
""")
