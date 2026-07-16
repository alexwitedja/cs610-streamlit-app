import streamlit as st

st.set_page_config(
    page_title="Model Comparisons",
    page_icon="⚖️",
)

st.title("Model Comparisons")

st.markdown("""
# What we tried, and what we chose
### And why the best model on paper is not the one you want in an ED

### The leaderboard

*(Render from `load_metrics()`, `variant == "raw"`, test set. Sortable. Highlight the winner per side.)*

**Patient-facing** — winner: **MLP deep ensemble**
\n**Hospital-facing** — winner: **MLP + XGBoost soft-voting ensemble**

Read this table with two things in mind.

**First — the extra clinical data buys less than you would think.** The hospital model sees full vitals,
arrival mode, and visit history that the patient model simply does not have. It converts that into a QWK
of 0.62 — against the patient model's own 0.63. The extra vitals, arrival context, and visit history don't
just fail to move the needle much; on this test set the leaner patient-facing model is marginally *ahead*,
a gap of about 0.002 in the other direction. The ClinicalBERT chief-complaint embedding is doing so much of
the work that a patient at home, with a phone and a thermometer, gets essentially all of the way to the
bedside model. **That is the most commercially interesting result in this project**, and it is the case for
the patient-facing app existing at all.

**Second — QWK and ESI-1 recall disagree, and that disagreement is the whole problem.** On the patient side,
the raw MLP takes the top QWK (0.63) while catching only **44.2%** of ESI-1 patients — worse on the class
that matters most than models it beats overall. The metric that looks like "best" is not the metric that
keeps people alive.

### The safety problem, stated plainly

Every raw model in this project misses roughly **a quarter of critical patients**. Pooled critical recall
(ESI 1–2 correctly kept in ESI 1–2) sits around 79.1% on the hospital side and 75.0% on the patient side.
Critical under-triage — a genuinely sick patient sent to the non-urgent queue — runs at
**20.9%** and **25.0%**.

For a decision-support tool, that is not an acceptable default. Argmax is the wrong decision rule when the
costs are asymmetric.

### The fix: a priority-threshold cascade

Rather than take the model's most likely class, we override it when the probability of a critical class
clears a tuned bar:

```
if P(ESI-1) ≥ t1:   predict ESI-1
elif P(ESI-2) ≥ t2: predict ESI-2
else:               predict argmax
```

`t1` and `t2` are chosen by grid search **on the validation set only** — maximise pooled critical recall,
subject to a floor of **QWK ≥ 0.55**, so safety cannot be bought by degrading the model into a
"scream ESI-1 at everything" alarm. The test set is touched once, at the end.

### What tuning actually costs

The cascade works, and it is not free:

**Patient-facing (MLP deep ensemble):**
- Pooled critical recall rises from **75.0%** to **89.4%**.
- Critical under-triage falls from **25.0%** to **10.6%** — roughly a **2.4× reduction** in the error that
  can kill someone.
- **And over-triage roughly doubles**, from **15.2%** to **31.0%**. QWK falls from 0.63 to 0.57.

**Hospital-facing (soft-voting ensemble):**
- Pooled critical recall rises from **79.1%** to **90.3%**.
- Critical under-triage falls from **20.9%** to **9.7%** — roughly a **2.2× reduction** in the error that
  can kill someone.
- **And over-triage roughly doubles**, from **17.5%** to **27.6%**. QWK falls from 0.62 to 0.59.

**This is the correct trade, and it is a choice, not an accident.** A model that is better on paper
(higher QWK, lower MAE) is worse in an emergency department, because the errors it saves are the cheap ones
and the errors it makes are the expensive ones. We report both operating points and let the reader see the
price. Someone deploying this would set that threshold with a clinical director in the room, not a data
scientist alone.

### Closing note — what we would do next

- **Validate locally.** MIMIC-IV-ED is one US hospital system; Singapore's case-mix is not the same.
- **Calibrate.** The reliability analysis shows the base models are not perfectly calibrated; isotonic
  regression per class recovered ESI-1 recall in earlier experiments at an explicit over-triage cost.
- **Watch the ESI-1 misses.** Error analysis says the ones we miss are the *atypical presentations* —
  critically ill patients whose vitals look normal. No amount of threshold tuning fixes that; it needs
  features we do not have.
""")