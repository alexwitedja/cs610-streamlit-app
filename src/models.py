"""Model wrappers for the KiasuCare app.

Every model takes the SAME input: the canonical raw frame from FrameBuilder (parquet schema,
raw values, 768 cls_* columns). Each then applies its own private transform, because the three
families were trained on genuinely different feature contracts:

    MLP     782/809 = [768 CLS | 14/41 tabular]   tabular standard-scaled, CLS left raw
    XGBoost 783/823 = [15/55 tabular | 768 CLS]   no transform at all
    LogReg   47/87  = [15/55 tabular | 32 PCA]    PCA on CLS, then scale all 47/87

Note the CLS block leads for the MLP and trails for the others. Selecting columns by name
rather than position is what keeps that straight.

All classes return probabilities as (n, 5) in ascending ESI order (column 0 = ESI-1), which is
what makes soft voting across families valid. predict() always returns ESI 1-5, so the
0-vs-1-indexing difference between the underlying estimators never escapes this module.

The MLP runs in NumPy against the folded .npz weights: torch is a ~300 MB dependency for what
is, at eval, three Linear+ReLU layers. Verified identical to the checkpoints in
admin/verify_models.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

N_CLASSES = 5

# Tabular columns each MLP was trained on, verbatim from submission/modelling/{side}/mlp.ipynb.
# The order is curated, not parquet order, and must not be sorted.
MLP_TABULAR_COLS = {
    "patient": [
        "gender_M", "age_at_visit",
        "temperature", "heartrate", "pain",
        "has_med_cardiac", "has_med_diabetes", "has_med_respiratory",
        "has_med_psych", "has_med_opioid", "has_med_gi",
        "has_med_thyroid", "has_med_anticonvulsant", "has_med_bloodthinner",
    ],
    "hospital": [
        "temperature", "heartrate", "pain", "resprate", "o2sat", "sbp", "dbp",
        "age_at_visit", "gender_M",
        "med_count_cardiac", "med_count_diabetes", "med_count_psych",
        "med_count_opioid", "med_count_gi", "med_count_thyroid",
        "med_count_bloodthinner", "has_polypharmacy",
        "arrival_hour", "arrival_dow", "is_weekend",
        "transport_WALK IN",
        "prior_ed_visits_1yr", "prior_admissions_1yr",
        "days_since_last_ed", "last_ed_admitted",
        "shock_index", "qsofa_partial", "pulse_pressure",
        "elderly", "very_elderly",
        "hypoxia", "severe_hypoxia",
        "tachycardia", "bradycardia", "tachypnea", "hypotension", "hypertensive_urg",
        "elderly_hypoxia", "elderly_tachycardia", "elderly_hypotension", "elderly_shock",
    ],
}

CLS_COLS = [f"cls_{i}" for i in range(768)]

# Selected on validation by maximising pooled critical recall subject to QWK >= 0.55.
# Source: submission/best_models_analysis/best_{side}_model.ipynb.
THRESHOLDS = {"patient": (0.20, 0.25), "hospital": (0.50, 0.25)}

# Soft-vote weights, per side: each base model's (val QWK x val critical recall), normalised at
# construction. Source: submission/modelling/{side}/ensemble.ipynb. These differ by side — the
# patient side leans MLP, the hospital side leans XGBoost — so they are not interchangeable.
ENSEMBLE_WEIGHTS = {
    "patient": {"mlp": 0.4639, "xgb": 0.4383},   # -> 0.514 / 0.486
    "hospital": {"mlp": 0.4735, "xgb": 0.4832},  # -> 0.495 / 0.505
}


class BaseModel(ABC):
    """Same input, same output shape, private transform."""

    def __init__(self, side: str) -> None:
        if side not in ("patient", "hospital"):
            raise ValueError(f"side must be 'patient' or 'hospital', got {side!r}")
        self.side = side

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """(n, 5) float32, ascending ESI: column 0 is ESI-1."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Raw argmax, as ESI 1-5. For the tuned operating point use cascade()."""
        return self.predict_proba(X).argmax(axis=1) + 1

    def _check(self, probs: np.ndarray, n: int) -> np.ndarray:
        if probs.shape != (n, N_CLASSES):
            raise RuntimeError(f"{type(self).__name__} returned {probs.shape}, expected {(n, 5)}")
        return probs.astype(np.float32)


class MLPEnsemble(BaseModel):
    """Deep ensemble of 5 MLPs (seeds 42/123/456/789/1024), averaged softmax.

    Transform: standard-scale the tabular block, leave CLS raw, concatenate [CLS | tab].
    """

    def __init__(self, npz_path: str | Path, scaler_path: str | Path, side: str, n_seeds: int = 5):
        super().__init__(side)
        self.tabular_cols = MLP_TABULAR_COLS[side]
        self.scaler = joblib.load(scaler_path)
        data = np.load(npz_path)
        self.seeds = [
            [
                (data[f"seed{s}_W{i}"], data[f"seed{s}_b{i}"])
                for i in range(4)
            ]
            for s in range(n_seeds)
        ]

        expected = self.scaler.n_features_in_
        if len(self.tabular_cols) != expected:
            raise ValueError(
                f"{side} MLP scaler expects {expected} tabular features, "
                f"but MLP_TABULAR_COLS has {len(self.tabular_cols)}"
            )

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        tab = self.scaler.transform(X[self.tabular_cols].to_numpy(dtype=np.float32))
        cls = X[CLS_COLS].to_numpy(dtype=np.float32)
        return np.concatenate([cls, tab], axis=1).astype(np.float32)

    @staticmethod
    def _forward(x: np.ndarray, layers) -> np.ndarray:
        """BatchNorm is folded into the weights at export, so this is Linear+ReLU x3, Linear."""
        for W, b in layers[:-1]:
            x = np.maximum(x @ W.T + b, 0.0)
        W, b = layers[-1]
        logits = x @ W.T + b
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        z = self._transform(X)
        probs = np.mean([self._forward(z, layers) for layers in self.seeds], axis=0)
        return self._check(probs, len(X))


class XGBoostModel(BaseModel):
    """Gradient boosting on the full raw feature set. classes_ is [0..4] -> ascending ESI."""

    def __init__(self, bundle_path: str | Path, side: str):
        super().__init__(side)
        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.feature_names].to_numpy(dtype=np.float32)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._check(self.model.predict_proba(self._transform(X)), len(X))


class LogisticRegressionModel(BaseModel):
    """Linear baseline. CLS is compressed 768 -> 32 by a fitted PCA, then everything is scaled.

    classes_ is [1..5], already ascending ESI, so no index shift is needed here.
    """

    def __init__(self, bundle_path: str | Path, side: str):
        super().__init__(side)
        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]
        self.pca = bundle["pca"]
        self.scaler = bundle["scaler"]
        self.feature_names = bundle["feature_names"]
        self.tabular_cols = [c for c in self.feature_names if not c.startswith("pca_")]

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        pca = self.pca.transform(X[CLS_COLS].to_numpy(dtype=np.float32))
        tab = X[self.tabular_cols].to_numpy(dtype=np.float32)
        return self.scaler.transform(np.concatenate([tab, pca], axis=1))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._check(self.model.predict_proba(self._transform(X)), len(X))


class SoftVotingEnsemble(BaseModel):
    """Weighted soft vote over base models that each own their transform.

    Composes on probabilities, not features: there is no single matrix that satisfies both the
    MLP and XGBoost, so the children are handed the same canonical frame and asked for probs.
    """

    def __init__(self, mlp: BaseModel, xgb: BaseModel, weights: dict[str, float] | None = None):
        if mlp.side != xgb.side:
            raise ValueError(f"side mismatch: mlp={mlp.side!r}, xgb={xgb.side!r}")
        super().__init__(mlp.side)
        self.mlp, self.xgb = mlp, xgb
        w = weights or ENSEMBLE_WEIGHTS[self.side]
        total = w["mlp"] + w["xgb"]
        self.w_mlp, self.w_xgb = w["mlp"] / total, w["xgb"] / total

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.w_mlp * self.mlp.predict_proba(X) + self.w_xgb * self.xgb.predict_proba(X)
        return self._check(probs, len(X))


def cascade(probs: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """Priority-threshold cascade -> ESI 1-5.

        if P(ESI-1) >= t1:   ESI-1
        elif P(ESI-2) >= t2: ESI-2
        else:                argmax

    Deliberately trades accuracy for critical recall. Kept outside the model classes because it
    is an operating-point choice, not part of the model: the same probabilities serve both the
    tuned and untuned views, so a UI toggle costs no extra inference.
    """
    preds = np.full(len(probs), -1, dtype=int)
    preds[probs[:, 0] >= t1] = 0
    preds[(preds == -1) & (probs[:, 1] >= t2)] = 1
    unresolved = preds == -1
    preds[unresolved] = probs[unresolved].argmax(axis=1)
    return preds + 1
