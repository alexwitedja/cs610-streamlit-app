# KiasuCare — Streamlit portfolio copy

**Rule enforced throughout:** every number below is written as a `{placeholder}` token. Nothing numeric is
hardcoded in the app. All figures resolve at runtime from **`metrics.csv`** — a single aggregate artifact
built offline by `admin/build_metrics.py`.

**The app never reads a `*_predictions.csv` or a `*.parquet`.** Those carry `subject_id` / `stay_id` and are
covered by the PhysioNet DUA. `build_metrics.py` reads them locally and emits aggregates only.

**Metrics contract** — `metrics.csv` is tidy long: `scope, side, model, variant, metric, value`, where
`side ∈ {patient, hospital}`, `model ∈ {lr, rf, xgb, mlp, ensemble}`, `variant ∈ {raw, tuned}`, and
`scope ∈ {model_metric, dataset, class_distribution, threshold, best_model, confusion}`.
The `value` column is mixed (numeric + a few string labels) — cast after filtering by scope.
Every `{placeholder}` in this document is a lookup into that frame.

---

## Page 1 — Welcome

### Hero

# KiasuCare
### Decision support for emergency department triage

An ordinal machine-learning system that predicts **ESI acuity (levels 1–5)** at the moment a patient
presents — built on {n_test:,} held-out emergency department visits from MIMIC-IV-ED.

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
the critical classes are rare. ESI-1 is only {pct_esi1}% of our test set. **Any honest triage model has to
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
| Feature count | ~783 | ~823 |
| Best test QWK | {qwk[patient,best,raw]} | {qwk[hospital,best,raw]} |

The gap between those two numbers is the entire value of the clinical data — and it is **smaller than you
would expect**. That finding is the point of the Cross-Case analysis.

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

---

## Page 2 — Inference

### Header

# Try it
Enter a presentation and see what KiasuCare predicts.

> Predictions are illustrative. This is a coursework model on US data — do not use it for real triage.

### Controls (copy for the widgets)

**Radio — "Which model?"**
- `Patient-facing` — What someone could self-report from home. Fewer vitals, no clinical context.
- `Hospital-facing` — What the ED knows at the bedside. Full vitals and arrival context.

*Helper text under the radio:* Switching sides changes the form, because the two models are given genuinely
different information. That is the only difference between them — the target is the same ESI 1–5 either way.

**Checkbox — "Apply clinical-safety threshold tuning"**

*Unchecked (raw):* The model's raw argmax — its single most likely class.
*Checked (tuned):* The priority cascade. If P(ESI-1) ≥ **{t1[side]}**, escalate to ESI-1. Otherwise if
P(ESI-2) ≥ **{t2[side]}**, escalate to ESI-2. Otherwise, argmax.

*Helper text under the checkbox:* Tuning is not free. On the {side} model it lifts pooled critical recall
from **{pooled_critical_recall[side,best,raw]}** to **{pooled_critical_recall[side,best,tuned]}** — but
over-triage rises from **{overtriage[side,best,raw]}** to **{overtriage[side,best,tuned]}** and QWK falls
from **{qwk[side,best,raw]}** to **{qwk[side,best,tuned]}**. You are buying safety with clinician time.

### Result panel

**Predicted acuity: ESI {pred}**
{esi_label[pred]} — *(1 = Resuscitation · 2 = Emergent · 3 = Urgent · 4 = Less urgent · 5 = Non-urgent)*

Show the full probability distribution across all five classes as a bar chart — **not just the winner.**
An ordinal model that is 45/40 split between ESI-2 and ESI-3 is telling you something a single number hides.

*When tuning has overridden the argmax, say so explicitly:*
> ⚠️ **Escalated by the safety rule.** Raw model said **ESI {raw_pred}**, but P(ESI-{k}) =
> **{p_k}** cleared the **{t_k}** threshold. This is the cascade doing its job — and it is exactly the
> kind of over-triage the tuned operating point accepts on purpose.

*Chief-complaint note:* The free text is embedded with ClinicalBERT at inference time. Cache the model —
loading it per-request will make the demo feel broken.

### Sidebar — "What this model was given"
List the fields actually fed to the model for the selected side, so the user can see the information
asymmetry rather than take it on trust.

---

## Page 3 — Data

### Header

# The data
### MIMIC-IV-ED, and the pipeline that makes it modellable

### Source

Four raw tables — `edstays`, `triage`, `patients`, `medrecon` — from **MIMIC-IV-ED**, a de-identified
record of emergency department visits at a US academic medical centre. Access is governed by a PhysioNet
data use agreement, so **no raw records are shown here** — only aggregates and the pipeline itself.

After cleaning: **{n_total:,} visits**, split **70/10/20** into train/validation/test
({n_train:,} / {n_val:,} / {n_test:,}).

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
| 1 | Resuscitation | {n_esi1:,} | {pct_esi1}% |
| 2 | Emergent | {n_esi2:,} | {pct_esi2}% |
| 3 | Urgent | {n_esi3:,} | {pct_esi3}% |
| 4 | Less urgent | {n_esi4:,} | {pct_esi4}% |
| 5 | Non-urgent | {n_esi5:,} | {pct_esi5}% |

Two consequences drive everything downstream. **ESI-3 dominates** — a model that predicts "3" every time
scores {pct_esi3}% accuracy while being clinically worthless, which is why accuracy is not our headline
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

---

## Page 4 — Model comparisons

### Header

# What we tried, and what we chose
### And why the best model on paper is not the one you want in an ED

### The leaderboard

*(Render from `load_metrics()`, `variant == "raw"`, test set. Sortable. Highlight the winner per side.)*

**Patient-facing** — winner: **MLP deep ensemble**
**Hospital-facing** — winner: **MLP + XGBoost soft-voting ensemble**

Read this table with two things in mind.

**First — the extra clinical data buys less than you would think.** The hospital model sees full vitals,
arrival mode, and visit history that the patient model simply does not have. It converts that into a QWK
of {qwk[hospital,best,raw]} against the patient model's {qwk[patient,best,raw]} — a gap of
{qwk_gap}. The ClinicalBERT chief-complaint embedding is doing so much of the work that a
patient at home, with a phone and a thermometer, gets most of the way to the bedside model. **That is the
most commercially interesting result in this project**, and it is the case for the patient-facing app
existing at all.

**Second — QWK and ESI-1 recall disagree, and that disagreement is the whole problem.** On the patient side,
the raw MLP takes the top QWK ({qwk[patient,mlp,raw]}) while catching only **{recall_esi1[patient,mlp,raw]}**
of ESI-1 patients — worse on the class that matters most than models it beats overall. The metric that looks
like "best" is not the metric that keeps people alive.

### The safety problem, stated plainly

Every raw model in this project misses roughly **a quarter of critical patients**. Pooled critical recall
(ESI 1–2 correctly kept in ESI 1–2) sits around {pooled_critical_recall[hospital,best,raw]} on the hospital
side and {pooled_critical_recall[patient,best,raw]} on the patient side. Critical under-triage —
a genuinely sick patient sent to the non-urgent queue — runs at
**{critical_undertriage[hospital,best,raw]}** and **{critical_undertriage[patient,best,raw]}**.

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

*(Render as a raw-vs-tuned delta table from `load_metrics()`, both variants, per side.)*

The cascade works, and it is not free:

- Pooled critical recall rises from **{pooled_critical_recall[side,best,raw]}** to
  **{pooled_critical_recall[side,best,tuned]}**.
- Critical under-triage falls from **{critical_undertriage[side,best,raw]}** to
  **{critical_undertriage[side,best,tuned]}** — roughly a **{crit_under_reduction}× reduction** in the
  error that can kill someone.
- **And over-triage roughly doubles**, from **{overtriage[side,best,raw]}** to
  **{overtriage[side,best,tuned]}**. QWK, MAE and accuracy all get *worse*.

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

---

## Page 5 — KakiCare (chat)

> **Design note, not app copy.** This page is the only one that *tells a member of the public what to do
> about their health*, so it is built to a different standard than the rest of the site.
>
> Three rules are non-negotiable, and each follows from a number in `metrics.csv`:
>
> 1. **The tuned model only.** No raw/tuned toggle here. Raw patient-side ESI-1 recall is
>    **{recall_esi1[patient,mlp,raw]}** — it misses more than half of the most critical patients. Exposing
>    that to a member of the public would be indefensible. The safety cascade is always on, silently.
> 2. **A red-flag rule layer runs BEFORE the model, and can only escalate.** Even tuned, pooled critical
>    recall is **{pooled_critical_recall[patient,mlp,tuned]}** — roughly **{n_missed_in_ten} in 10** critical
>    patients are still under-triaged. Error analysis says the ones we miss are *atypical presentations with
>    normal-looking vitals*, which is precisely what a lean feature set cannot see. Hard-coded red flags
>    (chest pain, breathing difficulty, stroke signs, uncontrolled bleeding, altered consciousness,
>    pregnancy complications, infant fever) bypass the model entirely and go straight to emergency advice.
> 3. **KakiCare never tells anyone to stay home.** The lowest-acuity result says "A&E is probably not the
>    fastest route for this — a GP or polyclinic likely can help," never "you don't need care." The model's
>    over-triage is cheap; its under-triage is not, and the copy is written to fail in the safe direction.

### Header

# KakiCare
### Your kaki for figuring out where to go

Tell me what's going on, and I'll give you a sense of how urgent it might be — and where to go for it.

> ⚠️ **KakiCare is a student project, not a doctor.** It runs on a model trained on US hospital records and
> has never been clinically validated. It can be wrong, and it is wrong most often about the patients who
> are sickest. **If you feel this is an emergency, call 995 now — don't chat with me.**

### The conversation (turn-by-turn copy)

**Turn 0 — open**
> Hi, I'm KakiCare 👋
>
> I'll ask you a few quick questions, then give you a rough sense of how urgent things look and what your
> options are. Takes about a minute.
>
> First — **what's bothering you today?** Describe it in your own words, like you'd tell a friend.

*(Free text → ClinicalBERT embedding. This is the single highest-signal input the model gets — the copy
should encourage detail, not a one-word answer.)*

**Red-flag interrupt** — fires immediately on the complaint text, before anything else:
> 🚨 **Stop — this needs emergency care now.**
>
> What you've described ({matched_flag}) can be life-threatening, and it's not something I should be
> triaging over chat.
>
> **Call 995 for an ambulance, or get to the nearest A&E immediately.**
>
> I'm not going to give you a score for this. Please go now.

*(No prediction is shown. No "but the model said ESI-3." The rule wins, always.)*

**Turn 1 — age and gender**
> Got it. How old are you, and what's your gender?

**Turn 2 — pain**
> On a scale of **0 to 10**, how bad is the pain right now? (0 = none, 10 = worst you can imagine)
>
> If it's not really a pain thing, just say **skip**.

*("skip" → `pain_missing = 1`. The flag is a real feature — a patient who can't give a pain score is
informative, and the pipeline models it that way. Don't silently impute a 0.)*

**Turn 3 — vitals**
> Two last things, if you can measure them:
>
> - **Temperature** (°C) — a thermometer reading if you have one
> - **Heart rate** (bpm) — most phones and watches can do this
>
> Don't have them? Say **skip** and I'll work with what I've got.

*(These are the only vitals the lean model gets. If skipped, they're imputed by the same train-fitted
KNN imputer used in the pipeline — say so in the "what I used" panel rather than hiding it.)*

**Turn 4 — medications**
> Last one. Are you on any regular medications? Rough categories are fine — heart, blood pressure, diabetes,
> blood thinners, painkillers, anything else. Or just say **none**.

### The result card

**Band A — ESI 1 or 2 (Emergent)**
> 🔴 **This looks urgent. Go to A&E now.**
>
> Based on what you've told me, your presentation looks like something that needs to be seen **immediately**.
>
> **Call 995 for an ambulance if you can't get there safely, or go straight to the nearest A&E.**
>
> Don't wait to see if it improves. If things get worse on the way, call 995.

**Band B — ESI 3 (Urgent)**
> 🟠 **You should be seen today.**
>
> This looks like something that needs proper medical attention, but not necessarily an ambulance.
>
> **Go to an A&E or an Urgent Care Centre today.** Bring a list of your medications if you can.
>
> If it gets noticeably worse while you're waiting — worse pain, trouble breathing, feeling faint —
> treat it as an emergency and call 995.

**Band C — ESI 4 or 5 (Less urgent)**
> 🟢 **This doesn't look like an emergency — but do get it looked at.**
>
> An A&E is probably not the fastest route for this; you'd likely wait a long time behind more urgent cases.
> **A GP or polyclinic is likely the better option**, and you'll be seen sooner.
>
> **This is not me telling you it's nothing.** I'm a model with a partial picture, and I get the sickest
> patients wrong more often than any other kind. If you feel worse than this makes it sound, trust that
> and get seen anyway.

*(Every band ends with an escalation path. There is no terminal "you're fine" state.)*

### Always-visible footer on the result

**How confident is this?**
Show the full 5-class probability bar chart, not just the winner. A 45/40 split between "urgent" and
"go today" is information the user deserves.

**What I actually knew about you**
List the fields collected, and — critically — the ones that were skipped and imputed. Honesty about the
model's blind spots is the feature, not a disclaimer.

**How often is KakiCare right?**
> On {n_test:,} held-out emergency visits, this model agrees with the triage nurse's exact score about
> **{accuracy[patient,mlp,tuned]}** of the time, and lands within one level about {within_one} of the time.
>
> The number that matters more: of patients who genuinely needed emergency care, it correctly flagged
> **{pooled_critical_recall[patient,mlp,tuned]}** of them — which means it **missed about
> {critical_undertriage[patient,mlp,tuned]}**. That is not good enough to rely on. It's good enough to be
> a second opinion, and that's all this is.

*(This is the most important paragraph on the page. Do not soften it, and do not hide it behind an
expander. A tool that tells people its own failure rate in plain language is more trustworthy, not less —
and it's the honest reading of {critical_undertriage[patient,mlp,tuned]}.)*

### Disclaimer strip (persistent, bottom of page)

> KakiCare is a CS610 student project built on MIMIC-IV-ED, a de-identified dataset from a **US** hospital
> system. It has not been validated on Singapore patients, has not been reviewed by any clinician, and is
> **not a medical device**. Nothing here is medical advice. In an emergency, call **995**.

---

### The US-data / Singapore-advice mismatch

*(Render as an always-visible callout on the KakiCare page — not tucked inside an expander. The short
version is the callout; the four numbered sections sit behind a "Why this matters" toggle.)*

#### Callout (always visible)

> ⚠️ **A model trained in America is giving you advice about Singapore.**
>
> KakiCare learned from **{n_total:,} emergency visits at a single US hospital system**, then hands you
> advice framed around 995, A&E, and polyclinics. Those two halves have never been joined up and tested.
> The seam between them is the least trustworthy part of this whole project — and it runs directly through
> the advice you just read.
>
> Read the number as **"roughly how urgent does this pattern look?"** — not as a triage category any
> Singapore hospital would recognise.

#### Why this matters (toggle)

**1. The scale itself is the wrong scale.**
KakiCare predicts **ESI (Emergency Severity Index), a 5-level US scale**, assigned by US nurses following US
protocols. Singapore's public EDs don't use ESI — they triage on **PACS (Patient Acuity Category Scale),
which has four levels (P1–P4)** and different decision rules for sorting patients into them.

These are not the same instrument with different labels, and **we have not built or validated a crosswalk
between them.** An earlier attempt at an ESI→PACS mapping was explored and never carried through, so we make
no claim about it. When KakiCare says "ESI 3," that does **not** mean "P3" — it means *the model thinks this
presentation resembles the US visits a US nurse labelled ESI 3*. The translation into a Singapore triage
category is a gap we have not closed, and pretending otherwise would be the single most misleading thing we
could do on this page.

**2. The people who show up are different people.**
A model's sense of "normal" is just the case-mix it was trained on, and ours is American. Two gaps matter
most.

*Who uses an ED at all.* In the US, emergency departments absorb a great deal of primary care — people
without a regular doctor use the ED as a first stop. Singapore has a dense **polyclinic and GP layer** that
catches most of those patients long before an A&E. So a whole population of low-acuity US visits — the ESI-4
and ESI-5 cases, **{pct_esi4}% and {pct_esi5}%** of our test data — arrives in an American ED but largely
never reaches a Singapore one. The model's prior over "how urgent is a typical arrival" is calibrated to the
wrong front door.

*Who the patients are.* Demographics, disease prevalence, and injury patterns differ between the two
populations. The model has learned relationships between vitals, complaints, and acuity from one of them.

**3. That shift breaks the probabilities, not just the label.**
This is subtler than "the accuracy might be lower," and it matters more. Every number KakiCare relies on —
the tuned thresholds **t1 = {t1[patient]}** and **t2 = {t2[patient]}**, the pooled critical recall of
**{pooled_critical_recall[patient,mlp,tuned]}** — was chosen against **US prevalence**. Thresholds are only
meaningful relative to the base rates they were tuned on. Move the model to a population where the mix of
arrivals is different, and the same threshold sits at a different point on the safety/over-triage curve.

The honest position: **we do not know what KakiCare's critical recall would be in a Singapore ED.** We know
what it is on held-out US patients. Those are different claims, and only one of them has evidence.

**4. The advice layer is ours, not the model's.**
The mapping from a predicted acuity to "call 995" / "go to A&E" / "see a GP or polyclinic" was written by us,
against Singapore's care pathways. **The model did not learn it and cannot vouch for it.** It is a reasonable
reading of what each acuity level implies — and it is exactly the kind of reasonable-looking assumption that
needs a clinician and local validation data before anyone acts on it.

#### What would fix this

Not a better model — **local data.** Validating KakiCare would mean re-fitting and re-tuning on Singapore ED
presentations labelled with PACS, re-selecting the thresholds against local prevalence, and having the advice
mapping reviewed by clinicians who actually run the triage desk. Until that happens, this page is a
demonstration of a method, not a tool. **Treat the acuity as a conversation starter with a real clinician —
never as a substitute for one.**
