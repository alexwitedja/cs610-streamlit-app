# KiasuCare

**Decision support for emergency department triage.**

An ordinal machine-learning system that predicts **ESI acuity (levels 1–5)** at the moment a
patient presents — trained on 80,984 held-out emergency department visits from MIMIC-IV-ED.
Shipped as a multi-page Streamlit app.

> KiasuCare is decision support, not a replacement for clinician triage. It is tuned to make the
> errors a triage nurse can live with, not to win an accuracy leaderboard.

## What it does

Emergency departments assign every arrival an **Emergency Severity Index** (ESI) score from 1
(resuscitate now) to 5 (non-urgent). The scale is ordinal and the errors are not symmetric:
under-triaging a critical patient can be fatal, over-triaging costs clinician time. A model that
optimises plain accuracy quietly trades the first kind of error for the second, because the
critical classes are rare (ESI-1 is 4.3% of the test set).

KiasuCare predicts acuity from vitals, demographics, medication history, and the free-text chief
complaint — encoded with a frozen **ClinicalBERT** model into a 768-dim embedding — then applies a
**priority-threshold cascade** on top of the model's raw probabilities to buy back critical recall
at an explicit, reported over-triage cost.

### One task, two feature sets

There is one prediction task (ordinal ESI 1–5) and two feature sets, differing in what information
is available, not in what they predict:

| | **Patient-facing** | **Hospital-facing** |
|---|---|---|
| Who provides the data | The patient, from home | The ED, at the bedside |
| Vitals | Temperature, heart rate, pain score | Full core vitals + respiratory rate, SpO₂, blood pressure |
| Context | Demographics, medications | + arrival transport, ED visit history, 19 derived clinical features |
| Chief complaint | ClinicalBERT embedding | ClinicalBERT embedding |
| Feature count | 15 + 768 | 55 + 768 |
| Best test QWK | 0.63 | 0.62 |

Four model families are trained identically on both feature sets — Logistic Regression, Random
Forest, XGBoost, and a 5-seed MLP deep ensemble — plus an MLP+XGBoost soft-voting ensemble.

## App pages

| Page | File | Status |
|---|---|---|
| Welcome | [Welcome.py](Welcome.py) | Project overview and methodology |
| Inference | [pages/1_🩺_Inference.py](pages/1_🩺_Inference.py) | Manual form — enter a presentation, see the prediction across all four models, toggle the safety cascade |
| Kiasu Care (chat) | [pages/2_❤️_Kiasu_Care.py](pages/2_❤️_Kiasu_Care.py) | Conversational front-end (Groq/Llama) that gathers a presentation turn-by-turn and calls the patient-facing model as a tool |
| Data | [pages/3_📊_Data.py](pages/3_📊_Data.py) | In progress |
| Model Comparisons | [pages/4_⚖️_Model_Comparisons.py](pages/4_⚖️_Model_Comparisons.py) | In progress |

## Architecture

```
src/
  embedder.py           ClinicalBERT [CLS] embedder — ONNX (fp16), no torch/transformers at inference
  frame_builder.py       Raw user input -> canonical feature frame (per side, matches training schema)
  models.py               Model wrappers: LogisticRegression, XGBoost, MLPEnsemble, SoftVotingEnsemble
  inference_pipeline.py  Wires embedder + frame builder + models together; the threshold cascade

pages/tools/             Tool-calling schema + executor the chat page exposes to the LLM
artifacts/
  embedder/               ONNX model + tokenizer (tracked with Git LFS)
  models/{patient,hospital}/  Per-side model weights
  inference_constants.json    Clip ranges, medians, schema — used to reproduce the training pipeline at inference
  metrics.csv             Aggregate offline metrics (no row-level patient data)
  prompts/                System/assistant prompts for the chat page
```

Every model consumes the same canonical raw frame from `FrameBuilder` and applies its own private
transform (the MLP scales+concatenates its tabular block with a raw CLS block; XGBoost takes the
full raw frame; Logistic Regression compresses CLS via a fitted PCA). All three return probabilities
as `(n, 5)` in ascending ESI order, which is what makes the soft-voting ensemble and the priority
cascade valid across model families. See the module docstrings in `src/` for the exact contracts.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The ONNX embedder (`artifacts/embedder/clinicalbert_cls_fp16.onnx`, ~215 MB) is tracked with
**Git LFS** — make sure `git lfs pull` has run if the file looks like a small pointer rather than
~200 MB.

The chat page (`pages/2_❤️_Kiasu_Care.py`) calls the Groq API and needs a key in
`.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "..."
```

## Running

```bash
uv run streamlit run Welcome.py
```

## Caveats

MIMIC-IV-ED is a single US hospital system. Singapore's ED case-mix, ESI application (Singapore
uses PACS, not ESI), and population differ, and the model's tuned thresholds were selected against
US prevalence. Nothing here is validated locally, it has not been reviewed by any clinician, and it
is **not a medical device** — it should not be deployed without local validation. In an emergency,
call **995**.

## Credits

Built as a personal portfolio project, based on work for SMU's CS610 (Applied Machine Learning) by
Alexander Witedja, Lilian Young, Tran Nguyen Ngoc Nhu, Li Tong Yang, and Ryan Edbert Nurtanio.
Thanks to [MIMIC](https://mimic.mit.edu/) for the dataset.
