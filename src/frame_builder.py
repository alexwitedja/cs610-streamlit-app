"""FrameBuilder — raw user input -> the canonical feature frame every model consumes.

One frame, built to the parquet schema with raw human-readable values (temperature in C,
heart rate in bpm, age in years) plus the 768 ClinicalBERT columns. Each model then applies
its own private transform: the MLP scales 14/41 tabular columns and concatenates [CLS | tab],
XGBoost takes all 783/823 raw in [tab | CLS] order, LogReg runs the CLS block through a fitted
PCA. So this class must emit every column any model might want, correctly named.

Every rule here is transcribed from submission/data_processing/preprocessing.ipynb. Training-only
steps (splitting, dropping EXPIRED rows, fitting the imputer, mode-filling acuity, leakage-safe
visit-history cumcounts) have no inference equivalent and are absent by design.

Deviation from the notebook, deliberate: skipped fields are median-filled rather than
KNN-imputed. sklearn's KNNImputer stores all 286,319 training rows in _fit_X, so shipping it
would blow the memory budget and redistribute row-level MIMIC data. Median fill ignores the
other features, which KNN would use. Disclose this in the UI.

This class fails silently if it drifts: a wrong unit or a mis-ordered column yields a confident,
plausible ESI rather than an error. It is verified against the test parquet in
admin/verify_frame_builder.py — keep that passing.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MED_CLASSES = [
    "cardiac", "diabetes", "respiratory", "psych", "opioid",
    "gi", "thyroid", "anticonvulsant", "bloodthinner",
]


class FrameBuilder:
    """Builds the canonical raw feature frame for one side ('patient' or 'hospital').

    The Embedder is injected so the builder can be tested with pre-computed CLS vectors,
    without loading a 400 MB transformer.
    """

    def __init__(self, embedder: Any, constants: dict, side: str) -> None:
        if side not in ("patient", "hospital"):
            raise ValueError(f"side must be 'patient' or 'hospital', got {side!r}")
        self.embedder = embedder
        self.side = side
        self.clips = {k: tuple(v) for k, v in constants["clips"].items()}
        self.medians = constants["medians"]
        self.age_range = tuple(constants["age_range_trained"])
        self.transport_categories = constants["transport_categories"]
        self.poly_threshold = constants["polypharmacy_threshold"]
        self.tabular_cols = constants["schema"][side]["tabular_cols"]
        self.cls_cols = [f"cls_{i}" for i in range(768)]
        self.columns = self.tabular_cols + self.cls_cols

    @classmethod
    def from_json(cls, embedder: Any, path: str | Path, side: str) -> FrameBuilder:
        return cls(embedder, json.loads(Path(path).read_text()), side)

    # ── public API ───────────────────────────────────────────────────────────────────────
    @staticmethod
    def fahrenheit_to_celsius(f: float) -> float:
        """MIMIC records temperature in F; the pipeline converts before clipping."""
        return (f - 32) * 5 / 9

    def build(
        self,
        presentations: dict | list[dict],
        cls_vectors: np.ndarray | None = None,
        strict_age: bool = True,
    ) -> pd.DataFrame:
        """Assemble the canonical frame.

        `cls_vectors` bypasses the embedder (used by the verification harness). `strict_age`
        rejects ages outside the trained range: MIMIC-IV-ED is adults-only, so a prediction
        for a child is pure extrapolation on the population where triage errors are least
        forgivable. Callers should catch this and refuse, not silently predict.
        """
        if isinstance(presentations, dict):
            presentations = [presentations]
        if not presentations:
            raise ValueError("no presentations given")

        rows = [self._tabular_row(p, strict_age) for p in presentations]
        df = pd.DataFrame(rows)

        df = self._fill(df)
        if self.side == "hospital":
            df = self._add_derived_features(df)  # notebook: only on fully-imputed frames

        if cls_vectors is None:
            cls_vectors = self.embedder.embed([p.get("chief_complaint", "") for p in presentations])
        cls = pd.DataFrame(np.asarray(cls_vectors, dtype=np.float32), columns=self.cls_cols)

        out = pd.concat([df.reset_index(drop=True), cls], axis=1)
        return self._finalise(out)

    # ── per-row assembly ─────────────────────────────────────────────────────────────────
    def _tabular_row(self, p: dict, strict_age: bool) -> dict:
        age = p.get("age")
        if age is None:
            raise ValueError("age is required")
        if strict_age and not (self.age_range[0] <= age <= self.age_range[1]):
            raise ValueError(
                f"age {age} is outside the trained range {self.age_range[0]:.0f}-"
                f"{self.age_range[1]:.0f}. MIMIC-IV-ED contains adults only; predicting here "
                f"would be extrapolation. Refuse and refer to care instead."
            )

        row: dict[str, Any] = {
            "gender_M": int(str(p.get("gender", "")).upper().startswith("M")),
            "age_at_visit": float(age),
        }
        row.update(self._vitals(p))
        row.update(self._medications(p))
        if self.side == "hospital":
            row.update(self._arrival(p))
            row.update(self._history(p))
        return row

    def _vitals(self, p: dict) -> dict:
        needed = ["temperature", "heartrate"] if self.side == "patient" else [
            "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"
        ]
        out: dict[str, Any] = {}
        for col in needed:
            out[col] = self._clip(col, p.get(col))

        pain = self._parse_pain(p.get("pain"))
        out["pain"] = pain
        # Flag BEFORE filling — distinguishes 'no pain reported' from 'pain = 0'.
        out["pain_missing"] = int(pd.isna(pain))
        return out

    def _clip(self, col: str, value: Any) -> float:
        """Out-of-range -> NaN -> filled later. The notebook clips to NaN, not to the bound:
        a heart rate of 900 is a typo, and pretending it is 300 invents data."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        value = float(value)
        lo, hi = self.clips[col]
        return value if lo <= value <= hi else np.nan

    @staticmethod
    def _parse_pain(value: Any) -> float:
        """Numeric, '4-6' range midpoints, or free text. Anything unparseable -> NaN."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        text = str(value).strip().lower()
        if text in ("", "skip", "none", "n/a", "na", "unable", "uta"):
            return np.nan
        if text == "denies":
            return 0.0
        if "-" in text:
            try:
                return float(np.mean([float(part) for part in text.split("-")]))
            except ValueError:
                return np.nan
        try:
            return float(np.clip(float(text), 0, 10))
        except ValueError:
            return np.nan

    def _medications(self, p: dict) -> dict:
        """Patient side gets has_* flags; hospital side gets counts.

        A user selects classes, not the ETC descriptions medrecon carries, so counts default
        to 1 per selected class. Callers with real counts should pass `med_counts`.
        """
        counts = p.get("med_counts")
        if counts is None:
            selected = [m.lower() for m in p.get("medications", [])]
            counts = {m: (1 if m in selected else 0) for m in MED_CLASSES}
        counts = {m: int(counts.get(m, 0)) for m in MED_CLASSES}
        total = int(p.get("med_count_total", sum(counts.values())))

        if self.side == "patient":
            return {f"has_med_{m}": int(counts[m] > 0) for m in MED_CLASSES}

        out: dict[str, Any] = {"med_count_total": total}
        out.update({f"med_count_{m}": counts[m] for m in MED_CLASSES})
        out["has_polypharmacy"] = int(total >= self.poly_threshold)
        out["med_count_total_log"] = float(np.log1p(total))
        return out

    def _arrival(self, p: dict) -> dict:
        """Accepts either an `arrival_time` datetime or explicit hour/dow/month fields."""
        if p.get("arrival_hour") is not None:
            hour, dow, month = int(p["arrival_hour"]), int(p["arrival_dow"]), int(p["arrival_month"])
        else:
            when = p.get("arrival_time") or datetime.now()
            if isinstance(when, str):
                when = datetime.fromisoformat(when)
            hour, dow, month = when.hour, when.weekday(), when.month

        out = {
            "arrival_hour": hour,
            "arrival_dow": dow,
            "is_weekend": int(dow >= 5),
            "arrival_month": month,
        }
        transport = str(p.get("arrival_transport", "UNKNOWN")).upper()
        if transport not in self.transport_categories:
            transport = "UNKNOWN"
        out.update({f"transport_{c}": int(c == transport) for c in self.transport_categories})
        return out

    def _history(self, p: dict) -> dict:
        return {
            col: (np.nan if p.get(col) is None else float(p[col]))
            for col in [
                "prior_ed_visits_total", "prior_ed_visits_1yr", "prior_admissions_1yr",
                "days_since_last_ed", "last_ed_admitted",
            ]
        }

    # ── fill, derive, validate ───────────────────────────────────────────────────────────
    def _fill(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].isna().any():
                if col not in self.medians:
                    raise KeyError(f"no median constant for '{col}' — cannot fill")
                df[col] = df[col].fillna(self.medians[col])
        return df

    @staticmethod
    def _add_derived_features(d: pd.DataFrame) -> pd.DataFrame:
        """Verbatim from preprocessing.ipynb _add_derived_features. 19 features."""
        d = d.copy()
        d["shock_index"] = (d["heartrate"] / d["sbp"].replace(0, np.nan)).clip(0, 5)
        d["map"] = (d["sbp"] + 2 * d["dbp"]) / 3
        d["pulse_pressure"] = d["sbp"] - d["dbp"]
        d["qsofa_partial"] = (d["resprate"] > 22).astype(int) + (d["sbp"] < 100).astype(int)
        d["elderly"] = (d["age_at_visit"] >= 65).astype(int)
        d["very_elderly"] = (d["age_at_visit"] >= 80).astype(int)
        d["pediatric"] = (d["age_at_visit"] < 18).astype(int)
        d["fever"] = (d["temperature"] > 38.3).astype(int)
        d["hypoxia"] = (d["o2sat"] < 94).astype(int)
        d["severe_hypoxia"] = (d["o2sat"] < 90).astype(int)
        d["tachycardia"] = (d["heartrate"] > 100).astype(int)
        d["bradycardia"] = (d["heartrate"] < 50).astype(int)
        d["tachypnea"] = (d["resprate"] > 20).astype(int)
        d["hypotension"] = (d["sbp"] < 90).astype(int)
        d["hypertensive_urg"] = (d["sbp"] > 180).astype(int)
        d["elderly_hypoxia"] = d["elderly"] * d["hypoxia"]
        d["elderly_tachycardia"] = d["elderly"] * d["tachycardia"]
        d["elderly_hypotension"] = d["elderly"] * d["hypotension"]
        d["elderly_shock"] = d["elderly"] * (d["shock_index"] > 1.0).astype(int)
        return d

    def _finalise(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise RuntimeError(f"FrameBuilder produced no value for: {missing[:8]}")

        df = df[self.columns].astype(np.float32)
        if df.isna().to_numpy().any():
            bad = df.columns[df.isna().any()].tolist()
            raise RuntimeError(f"NaNs survived into the frame: {bad}")
        return df
