from __future__ import annotations

import json
from pathlib import Path

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
    constants = json.loads(ARTIFACTS_DIR / "inference_constants.json")
    return FrameBuilder(embedder, constants, side)
