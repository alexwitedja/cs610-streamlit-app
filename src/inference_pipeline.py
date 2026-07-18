from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

import streamlit as st

from src.embedder import ClinicalBertEmbedder
from src.frame_builder import FrameBuilder
from src.models import LogisticRegressionModel, MLPEnsemble, SoftVotingEnsemble, XGBoostModel

ARTIFACTS_DIR = Path().resolve() / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"


@st.cache_resource
def load_embedder():
    return ClinicalBertEmbedder(ARTIFACTS_DIR / "embedder", "clinicalbert_cls_fp16.onnx")


@st.cache_resource
def load_models(side: str):
    side_dir = MODELS_DIR / side

    xgb = XGBoostModel(side_dir / "xgb.ubj", side)
    mlp = MLPEnsemble(side_dir / "mlp_ensemble.npz", side_dir / "mlp_scaler.pkl", side)
    lr = LogisticRegressionModel(side_dir / "lr.pkl", side)
    return lr, xgb, mlp, SoftVotingEnsemble(mlp, xgb)


@st.cache_resource
def load_frame_builder(side: str):
    embedder = load_embedder()
    constants = json.loads((ARTIFACTS_DIR / "inference_constants.json").read_text())
    return FrameBuilder(embedder, constants, side)


def single_inference(raw_data: pd.DataFrame, side: str):
    models = load_models(side)
    frame_builder = load_frame_builder(side)
    X = frame_builder.build(raw_data)
    result = {}
    for model in models:
        probs = model.predict_proba(X)
        result[str(model)] = probs.flatten()

    return result

def cascade(probs: np.ndarray, t1: float = 0.20, t2: float = 0.25) -> np.ndarray:
    """Priority-threshold cascade -> ESI 1-5.

        if P(ESI-1) >= t1:   ESI-1
        elif P(ESI-2) >= t2: ESI-2
        else:                argmax

    Deliberately trades accuracy for critical recall. Kept outside the model classes because it
    is an operating-point choice, not part of the model: the same probabilities serve both the
    tuned and untuned views, so a UI toggle costs no extra inference.
    """
    probs = np.atleast_2d(probs)
    preds = np.full(len(probs), -1, dtype=int)
    preds[probs[:, 0] >= t1] = 0
    preds[(preds == -1) & (probs[:, 1] >= t2)] = 1
    unresolved = preds == -1
    preds[unresolved] = probs[unresolved].argmax(axis=1)
    return preds + 1
