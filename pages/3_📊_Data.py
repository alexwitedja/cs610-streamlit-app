import streamlit as st

st.set_page_config(
    page_title="Data",
    page_icon="📊",
)

st.title("The Data")

st.sidebar.header("Data")

st.markdown("""
### MIMIC-IV-ED, and the pipeline that makes it modellable

### Source

Four raw tables — `edstays`, `triage`, `patients`, `medrecon` — from **MIMIC-IV-ED**, a de-identified
record of emergency department visits at a US academic medical centre. Access is governed by a PhysioNet
data use agreement, so **no raw records are shown here** — only aggregates and the pipeline itself.

After cleaning: **408,088 visits**, split **70/10/20** into train/validation/test
(286,319 / 40,785 / 80,984).

### The split is by *patient*, not by visit

This is the detail that separates an honest evaluation from an inflated one. Patients return to the ED.
If you split randomly by visit, the same person's January visit lands in train and their March visit in
test — and the model gets to memorise the patient rather than learn the medicine.

KiasuCare splits on **`subject_id`**, stratified on each patient's **most-severe (minimum) acuity**, so a
given patient appears in exactly one split. Every number on this site is measured on patients the model
has **never seen**.

### The target is imbalanced, and ordinal

ESI distribution across the test set:

| ESI | Meaning | Test visits | Share |
|---|---|---|---|
| 1 | Resuscitation | 3,484 | 4.3% |
| 2 | Emergent | 27,020 | 33.4% |
| 3 | Urgent | 44,483 | 54.9% |
| 4 | Less urgent | 5,752 | 7.1% |
| 5 | Non-urgent | 245 | 0.3% |

Two consequences drive everything downstream. **ESI-3 dominates** — a model that predicts "3" every time
scores 54.9% accuracy while being clinically worthless, which is why accuracy is not our headline
metric. And **the classes are ordered** — mistaking ESI-1 for ESI-5 is far worse than mistaking it for
ESI-2 — which is why we lead with **QWK** and **MAE**, both of which penalise a miss by *how far* it missed.

### Handling missing values

Missingness in an ED is not random, and the pipeline treats it that way:

- **Rows with 4+ missing core vitals are dropped** — too little signal to impute honestly.
- **Out-of-range vitals are clipped to NaN**, not to the boundary. A recorded heart rate of 900 is a typo,
  not a tachycardic patient; pretending it is 300 invents data.
- **Continuous vitals and age are KNN-imputed**, with the imputer and the `StandardScaler` **fit on the
  training split only** and applied to val/test. Fitting on the full dataset leaks test statistics into
  training and is the most common way this pipeline could have been quietly wrong.
- **Missingness itself is a feature.** `pain_missing`, `cc_missing`, and `acuity_missing` are kept as
  flags — a patient too unwell to report a pain score is telling you something.
- **Missing acuity is mode-filled (mode = 3) *after* the split**, using the training mode, never the global one.

### Feature engineering

- **Chief complaint → 768-dim ClinicalBERT embedding.** The frozen `[CLS]` vector from a clinically
  pretrained BERT. This is the largest block of features and, per our diagnostics, the main driver of
  ESI-2 discrimination.
- **19 derived clinical features** from imputed vitals — shock index and related physiological ratios that
  encode interactions a tree would otherwise have to discover from scratch.
- **ED visit history**, computed leakage-safely with `cumcount`/`shift` so a visit only ever sees the
  patient's **prior** visits, never its own or future ones.
- **Medication features** from `medrecon` — counts, class rollups, and binary `has_*` flags.
- **Arrival context** — hour, day of week, weekend flag, month, and transport mode one-hots.
- **Demographics** — age at visit, gender, race one-hots.

### Temperature, °F → °C
A small but real one: MIMIC records temperature in Fahrenheit. It is converted before clipping, because a
"normal range" clip applied to the wrong unit would silently null out the entire column.

### Validation
The pipeline ends in hard assertions, not vibes: no patient crosses splits, acuity ∈ {1..5}, no NaNs
survive into the model matrix, and both feature sets match their declared contract. If any of those fail,
the notebook stops.
""")