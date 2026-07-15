import streamlit as st

st.set_page_config(
    page_title="Triage support system",
    page_icon="",
)

st.title("Triage support system")

st.sidebar.success("Navigate the app above.")

st.markdown("""
# KiasuCare
### Decision support for emergency department triage

An ordinal machine-learning system that predicts **ESI acuity (levels 1–5)** at the moment a patient
presents — built on 80,984 held-out emergency department visits from MIMIC-IV-ED.

> KiasuCare is **decision support, not a replacement for clinician triage.** Every design choice below
> follows from that: the system is tuned to make the errors a triage nurse can live with, not to win an
> accuracy leaderboard.

### The problem

Emergency departments run on triage. A nurse has minutes to assign each arrival an **Emergency Severity
Index** score from 1 (resuscitate now) to 5 (non-urgent), and that number decides who waits and who is
seen. The scale is ordinal, and the errors are not symmetric:

- **Under-triage** — calling a critically ill patient non-urgent — can be fatal.
- **Over-triage** — escalating someone who could have waited — costs clinician time and bed capacity.

A model that optimises plain accuracy will quietly trade the first kind of error for the second, because
the critical classes are rare. ESI-1 is only 4.3% of our test set. **Any honest triage model has to
be judged on how it fails, not just on how often it is right.**

### The solution

KiasuCare predicts acuity from what is known at the front door: vitals, demographics, medication history,
and — importantly — the **free-text chief complaint**, encoded with a frozen **ClinicalBERT** model into a
768-dimensional embedding. That text is what carries the clinical signal a vitals table misses ("crushing
chest pain radiating to jaw" is not a number).

Two deliberate failure-mode controls sit on top:

1. **A priority-threshold cascade** that overrides the model's argmax when the probability of a critical
   class clears a tuned bar — buying back critical recall at a stated over-triage cost.
2. **Explicit reporting of the trade**, so the operating point is a decision someone makes, not a default
   they inherit.

### One task, two feature sets

This is the single most important thing to understand about the project. There is **one prediction task** —
ordinal ESI 1–5 — and **two feature sets**, which differ in what information is available, not in what they
predict:

| | **Patient-facing** | **Hospital-facing** |
|---|---|---|
| Who provides the data | The patient, from home, on a phone | The ED, at the bedside |
| Vitals | Temperature, heart rate, pain score | Full core vitals + respiratory rate, SpO₂, blood pressure |
| Context | Demographics, medications | + arrival transport, ED visit history, 19 derived clinical features |
| Chief complaint | ✅ ClinicalBERT embedding | ✅ ClinicalBERT embedding |
| Feature count | 15 + 768 (text embeddings) | 55 + 768 (text embeddings) |
| Best test QWK | 0.63 | 0.62 |

The gap between those two numbers is the entire value of the clinical data — and it is **smaller than you
would expect** (the patient-facing model is, if anything, marginally ahead). That finding is the point of
the Model Comparisons page.

### Models evaluated

Four families, trained identically on both feature sets so the comparison is apples-to-apples, plus a
soft-voting ensemble:

- **Logistic Regression** — the linear baseline.
- **Random Forest** — bagged trees.
- **XGBoost** — gradient boosting; the strongest single tabular model.
- **MLP deep ensemble** — 5 neural nets (seeds 42/123/456/789/1024) with averaged softmax.
- **Soft-voting ensemble** — MLP + XGBoost probabilities combined.

**Winners:** the **MLP deep ensemble** on the patient side, and the **MLP + XGBoost soft-voting ensemble**
on the hospital side.

### Caveat, stated up front

MIMIC-IV-ED is a single US hospital system. Singapore's ED case-mix, ESI application, and population differ.
Nothing here is validated locally, and it should not be deployed without that validation.

*I built this streamlit app for my personal portfolio. It is based on a project for SMU's CS610 (Applied Machine Learning) built by Alexander Witedja, Lilian Young, Tran Nguyen Ngoc Nhu, Li Tong Yang, Ryan Edbert Nurtanio. Special thanks to [MIMIC](https://mimic.mit.edu/) for providing the dataset. Source code: https://github.com/alexwitedja/cs610-streamlit-app.*            
""")