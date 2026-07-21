from src.inference_pipeline import app_inference, cascade

MED_LABELS = {
    "diabetes": "diabetes", "cardiac": "cardiac", "respiratory": "respiratory",
    "psych": "psych", "opioid": "opioid",
    "anticonvulsant": "anticonvulsant", "bloodthinner": "bloodthinner",
    "thyroid": "thyroid", "digestive": "gi",
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
        "medications": [MED_LABELS[m.lower()] for m in medications] if medications else [],
    }

    return app_inference(X)
