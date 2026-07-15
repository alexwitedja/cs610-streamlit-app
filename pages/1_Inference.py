import streamlit as st

st.set_page_config(
    page_title="Inference",
    page_icon="",
)

st.title("Try it")
st.markdown("Enter a presentation and see what KiasuCare predicts.")

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

if mode == "Patient-facing":
    with st.form(key="Patient-facing"):
        st.subheader("Patient-facing inference")
        st.text_area("Chief complaint")
        st.form_submit_button("Run", type="primary")
else:
    with st.form(key="Hospital_facing"):
        st.subheader("Hospital-facing inference")
        st.text_area("Chief complaint")
        st.form_submit_button("Run", type="primary")

st.sidebar.caption(
    "Switching sides changes the form, because the two models are given genuinely different information. "
    "That is the only difference between them — the target is the same ESI 1–5 either way."
)

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

if mode == "Patient-facing":
    pass
else:
    pass

st.markdown("""
### Result

Predicted acuity, and how confident the model is, will render here once a presentation is entered:

**Predicted acuity: ESI `{pred}`**
`{esi_label}` — *(1 = Resuscitation · 2 = Emergent · 3 = Urgent · 4 = Less urgent · 5 = Non-urgent)*

The full probability distribution across all five classes is shown as a bar chart — not just the winner.
An ordinal model that is 45/40 split between ESI-2 and ESI-3 is telling you something a single number hides.

When tuning has overridden the argmax, this is stated explicitly:
> ⚠️ **Escalated by the safety rule.** Raw model said **ESI `{raw_pred}`**, but P(ESI-`{k}`) = **`{p_k}`**
> cleared the **`{t_k}`** threshold. This is the cascade doing its job — and it is exactly the kind of
> over-triage the tuned operating point accepts on purpose.
""")

st.caption(
    "The free text chief complaint is embedded with ClinicalBERT at inference time. The model is cached "
    "so it only loads once — loading it per-request would make the demo feel broken."
)

st.sidebar.markdown("### What this model was given")
if mode == "Patient-facing":
    st.sidebar.markdown("""
- Temperature, heart rate, pain score
- Demographics (age, gender, race)
- Medication history
- Chief complaint (ClinicalBERT embedding)
- ~783 features total
""")
else:
    st.sidebar.markdown("""
- Full core vitals + respiratory rate, SpO₂, blood pressure
- Demographics (age, gender, race)
- Medication history
- Arrival transport, ED visit history, 19 derived clinical features
- Chief complaint (ClinicalBERT embedding)
- ~823 features total
""")
