import streamlit as st

st.set_page_config(
    page_title="Kaki Care",
    page_icon="",
)

# Design note: this is the only page that tells a member of the public what to do about their health,
# so it holds itself to a different standard than the rest of the site.
#   1. The tuned model only — no raw/tuned toggle. Raw patient-side ESI-1 recall is 44.2%; exposing
#      that to the public would be indefensible. The safety cascade is always on, silently.
#   2. A red-flag rule layer runs BEFORE the model and can only escalate. Even tuned, pooled critical
#      recall is 89.4% — roughly 1 in 10 critical patients are still under-triaged. Hard-coded red flags
#      (chest pain, breathing difficulty, stroke signs, uncontrolled bleeding, altered consciousness,
#      pregnancy complications, infant fever) bypass the model entirely.
#   3. KakiCare never tells anyone to stay home. The lowest-acuity result says a GP/polyclinic is
#      probably faster, never "you don't need care."

st.title("KakiCare")
st.markdown("### Your kaki for figuring out where to go")
st.markdown(
    "Tell me what's going on, and I'll give you a sense of how urgent it might be — and where to go for it."
)
st.warning(
    "⚠️ **KakiCare is a student project, not a doctor.** It runs on a model trained on US hospital records "
    "and has never been clinically validated. It can be wrong, and it is wrong most often about the "
    "patients who are sickest. **If you feel this is an emergency, call 995 now — don't chat with me.**"
)

st.sidebar.header("Kaki Care")

st.divider()

st.markdown("### The conversation")
st.caption("A demonstration of how a chat interface would collect information and advise the user where to go.")

with st.expander("Turn 0 — open", expanded=True):
    st.markdown("""
Hi, I'm KakiCare 👋

I'll ask you a few quick questions, then give you a rough sense of how urgent things look and what your
options are. Takes about a minute.

First — **what's bothering you today?** Describe it in your own words, like you'd tell a friend.
""")
    st.caption(
        "Free text → ClinicalBERT embedding. This is the single highest-signal input the model gets — "
        "encourage detail, not a one-word answer."
    )

st.error("""
🚨 **Stop — this needs emergency care now.**

What you've described (`{matched_flag}`) can be life-threatening, and it's not something I should be
triaging over chat.

**Call 995 for an ambulance, or get to the nearest A&E immediately.**

I'm not going to give you a score for this. Please go now.
""")
st.caption("Red-flag interrupt — fires immediately on the complaint text, before anything else. No "
           "prediction is shown, and no model score overrides it. The rule wins, always.")

with st.expander("Turn 1 — age and gender"):
    st.markdown("Got it. How old are you, and what's your gender?")

with st.expander("Turn 2 — pain"):
    st.markdown("""
On a scale of **0 to 10**, how bad is the pain right now? (0 = none, 10 = worst you can imagine)

If it's not really a pain thing, just say **skip**.
""")
    st.caption(
        "\"skip\" → pain_missing = 1. A patient who can't give a pain score is informative — don't "
        "silently impute a 0."
    )

with st.expander("Turn 3 — vitals"):
    st.markdown("""
Two last things, if you can measure them:

- **Temperature** (°C) — a thermometer reading if you have one
- **Heart rate** (bpm) — most phones and watches can do this

Don't have them? Say **skip** and I'll work with what I've got.
""")
    st.caption(
        "These are the only vitals the lean model gets. If skipped, they're imputed by the same "
        "train-fitted KNN imputer used in the pipeline — that's disclosed in \"what I actually knew "
        "about you\" below, not hidden."
    )

with st.expander("Turn 4 — medications"):
    st.markdown(
        "Last one. Are you on any regular medications? Rough categories are fine — heart, blood "
        "pressure, diabetes, blood thinners, painkillers, anything else. Or just say **none**."
    )

st.divider()
st.markdown("### The result")

st.error("""
**Band A — ESI 1 or 2 (Emergent)**

🔴 **This looks urgent. Go to A&E now.**

Based on what you've told me, your presentation looks like something that needs to be seen **immediately**.

**Call 995 for an ambulance if you can't get there safely, or go straight to the nearest A&E.**

Don't wait to see if it improves. If things get worse on the way, call 995.
""")

st.warning("""
**Band B — ESI 3 (Urgent)**

🟠 **You should be seen today.**

This looks like something that needs proper medical attention, but not necessarily an ambulance.

**Go to an A&E or an Urgent Care Centre today.** Bring a list of your medications if you can.

If it gets noticeably worse while you're waiting — worse pain, trouble breathing, feeling faint —
treat it as an emergency and call 995.
""")

st.success("""
**Band C — ESI 4 or 5 (Less urgent)**

🟢 **This doesn't look like an emergency — but do get it looked at.**

An A&E is probably not the fastest route for this; you'd likely wait a long time behind more urgent cases.
**A GP or polyclinic is likely the better option**, and you'll be seen sooner.

**This is not me telling you it's nothing.** I'm a model with a partial picture, and I get the sickest
patients wrong more often than any other kind. If you feel worse than this makes it sound, trust that
and get seen anyway.
""")
st.caption("Every band ends with an escalation path. There is no terminal \"you're fine\" state.")

st.divider()
st.markdown("#### How confident is this?")
st.caption(
    "The full 5-class probability bar chart is shown here, not just the winner. A 45/40 split between "
    "\"urgent\" and \"go today\" is information the user deserves."
)

st.markdown("#### What I actually knew about you")
st.caption(
    "Lists the fields collected, and — critically — the ones that were skipped and imputed. Honesty "
    "about the model's blind spots is the feature, not a disclaimer."
)

st.markdown("#### How often is KakiCare right?")
st.markdown("""
On 80,984 held-out emergency visits, this model agrees with the triage nurse's exact score about
**57.3%** of the time, and lands within one level about **97.3%** of the time.

The number that matters more: of patients who genuinely needed emergency care, it correctly flagged
**89.4%** of them — which means it **missed about 10.6%**. That is not good enough to rely on. It's
good enough to be a second opinion, and that's all this is.
""")

st.divider()
st.caption(
    "KakiCare is a CS610 student project built on MIMIC-IV-ED, a de-identified dataset from a **US** "
    "hospital system. It has not been validated on Singapore patients, has not been reviewed by any "
    "clinician, and is **not a medical device**. Nothing here is medical advice. In an emergency, call **995**."
)

st.divider()
st.warning("""
⚠️ **A model trained in America is giving you advice about Singapore.**

KakiCare learned from **408,088 emergency visits at a single US hospital system**, then hands you advice
framed around 995, A&E, and polyclinics. Those two halves have never been joined up and tested. The seam
between them is the least trustworthy part of this whole project — and it runs directly through the
advice you just read.

Read the number as **"roughly how urgent does this pattern look?"** — not as a triage category any
Singapore hospital would recognise.
""")

with st.expander("Why this matters"):
    st.markdown("""
**1. The scale itself is the wrong scale.**
KakiCare predicts **ESI (Emergency Severity Index), a 5-level US scale**, assigned by US nurses following
US protocols. Singapore's public EDs don't use ESI — they triage on **PACS (Patient Acuity Category
Scale), which has four levels (P1–P4)** and different decision rules for sorting patients into them.

These are not the same instrument with different labels, and **we have not built or validated a crosswalk
between them.** An earlier attempt at an ESI→PACS mapping was explored and never carried through, so we
make no claim about it. When KakiCare says "ESI 3," that does **not** mean "P3" — it means *the model
thinks this presentation resembles the US visits a US nurse labelled ESI 3*. The translation into a
Singapore triage category is a gap we have not closed, and pretending otherwise would be the single most
misleading thing we could do on this page.

**2. The people who show up are different people.**
A model's sense of "normal" is just the case-mix it was trained on, and ours is American. Two gaps matter
most.

*Who uses an ED at all.* In the US, emergency departments absorb a great deal of primary care — people
without a regular doctor use the ED as a first stop. Singapore has a dense **polyclinic and GP layer**
that catches most of those patients long before an A&E. So a whole population of low-acuity US visits —
the ESI-4 and ESI-5 cases, **7.1% and 0.3%** of our test data — arrives in an American ED but largely
never reaches a Singapore one. The model's prior over "how urgent is a typical arrival" is calibrated to
the wrong front door.

*Who the patients are.* Demographics, disease prevalence, and injury patterns differ between the two
populations. The model has learned relationships between vitals, complaints, and acuity from one of them.

**3. That shift breaks the probabilities, not just the label.**
This is subtler than "the accuracy might be lower," and it matters more. Every number KakiCare relies on —
the tuned thresholds **t1 = 0.20** and **t2 = 0.25**, the pooled critical recall of **89.4%** — was chosen
against **US prevalence**. Thresholds are only meaningful relative to the base rates they were tuned on.
Move the model to a population where the mix of arrivals is different, and the same threshold sits at a
different point on the safety/over-triage curve.

The honest position: **we do not know what KakiCare's critical recall would be in a Singapore ED.** We
know what it is on held-out US patients. Those are different claims, and only one of them has evidence.

**4. The advice layer is ours, not the model's.**
The mapping from a predicted acuity to "call 995" / "go to A&E" / "see a GP or polyclinic" was written by
us, against Singapore's care pathways. **The model did not learn it and cannot vouch for it.** It is a
reasonable reading of what each acuity level implies — and it is exactly the kind of reasonable-looking
assumption that needs a clinician and local validation data before anyone acts on it.

**What would fix this**

Not a better model — **local data.** Validating KakiCare would mean re-fitting and re-tuning on Singapore
ED presentations labelled with PACS, re-selecting the thresholds against local prevalence, and having the
advice mapping reviewed by clinicians who actually run the triage desk. Until that happens, this page is a
demonstration of a method, not a tool. **Treat the acuity as a conversation starter with a real clinician —
never as a substitute for one.**
""")
