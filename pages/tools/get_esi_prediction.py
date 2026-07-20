from src.inference_pipeline import load_models, cascade

MED_LABELS = {
    "Diabetes": "diabetes", "Cardiac": "cardiac", "Respiratory": "respiratory",
    "Mental health, sleep, or anxiety": "psych", "Opioid": "opioid",
    "Anticonvulsant": "anticonvulsant", "Bloodthinner": "bloodthinner",
    "Thyroid": "thyroid", "Digestive": "gi",
}

def get_esi_prediction(
        chief_complaint,
        age,
        gender,
        pain,
        temperature,
        heartrate,
        medications,
):
    X = {
        "chief_complaint": chief_complaint,
        "age": age,
        "gender": "M" if gender == "Male" else "F",
        "temperature": temperature,
        "heartrate": heartrate,
        "pain": pain,
        "medications": [MED_LABELS[m] for m in medications],
    }

    _, _, mlp, _ = load_models("patient")

    probs = mlp.predict_proba(X)
    return cascade(probs)
