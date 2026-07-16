"""ClinicalBERT [CLS] embedder for the KiasuCare app.

Produces the 768-dim chief-complaint vectors the models were trained on, without importing
torch or transformers. Runs the fp16 ONNX export of emilyalsentzer/Bio_ClinicalBERT through
onnxruntime (~410 MB resident, ~20 ms per complaint).

Why fp16 and not int8: int8 dynamic quantisation changes the predicted ESI for ~7% of
patients and flips the critical band (ESI 1-2 vs 3-5) for ~3.5%, even with per-channel
scales. fp16 agrees with the notebook's fp32 embeddings on 99.93% of predictions.
See admin/tune_quant.py for the measurements.

Contract reproduced verbatim from submission/data_processing/clinicaBERT.ipynb:
    cleaning   : lower -> strip repeated underscores -> drop non-alphanumerics -> collapse spaces
    tokenising : padding='max_length', truncation=True, max_length=32
    output     : last_hidden_state[:, 0, :] -> the [CLS] vector

Any drift in the cleaning or max_length silently produces valid-looking but wrong vectors,
so both live here and nowhere else.

Usage:
    embedder = ClinicalBertEmbedder(MODEL_DIR)          # wrap in st.cache_resource
    cls = embedder.embed("crushing chest pain")          # -> (1, 768) float32
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

_MULTI_UNDERSCORE = re.compile(r"_{2,}")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")


class ClinicalBertEmbedder:
    """Frozen Bio_ClinicalBERT, emitting [CLS] vectors for chief-complaint text.

    Stateless after construction and safe to share across Streamlit sessions: the tokenizer's
    padding/truncation is configured once here, and both encode_batch and InferenceSession.run
    are re-entrant. Do not mutate the tokenizer after construction.
    """

    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    MAX_LENGTH = 32
    DIM = 768

    def __init__(
        self,
        model_dir: str | Path,
        onnx_filename: str = "clinicalbert_cls_fp16.onnx",
        intra_op_threads: int | None = None,
    ) -> None:
        model_dir = Path(model_dir)
        onnx_path = model_dir / onnx_filename
        tokenizer_path = model_dir / "tokenizer" / "tokenizer.json"

        for path in (onnx_path, tokenizer_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Embedder artifact missing: {path}. Export it with "
                    f"admin/export_embedder.py, or check the LFS files were pulled "
                    f"(a stale LFS pointer is a few hundred bytes, not a few hundred MB)."
                )

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=self.MAX_LENGTH)
        self._tokenizer.enable_padding(length=self.MAX_LENGTH)

        options = ort.SessionOptions()
        options.log_severity_level = 3
        if intra_op_threads is not None:
            options.intra_op_num_threads = intra_op_threads

        self._session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )

    @staticmethod
    def clean(text: str | float | None) -> str:
        """Verbatim from clinicaBERT.ipynb. Missing text embeds as '', exactly as in training."""
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return ""
        text = str(text).lower()
        text = _MULTI_UNDERSCORE.sub(" ", text)
        text = _NON_ALNUM.sub("", text)
        return _WHITESPACE.sub(" ", text).strip()

    def embed(self, texts: str | list[str], batch_size: int = 64) -> np.ndarray:
        """Embed one or more chief complaints. Returns (n, 768) float32.

        Cleaning is applied here so callers cannot forget it; it is idempotent, so passing
        already-cleaned text is harmless.
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, self.DIM), dtype=np.float32)

        cleaned = [self.clean(t) for t in texts]
        chunks = [
            self._run(cleaned[i : i + batch_size]) for i in range(0, len(cleaned), batch_size)
        ]
        out = np.concatenate(chunks, axis=0)

        if out.shape != (len(texts), self.DIM):
            raise RuntimeError(f"Embedder returned {out.shape}, expected {(len(texts), self.DIM)}")
        return out

    def _run(self, cleaned: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(cleaned)
        inputs = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        return self._session.run(["cls"], inputs)[0].astype(np.float32)

    def cls_columns(self) -> list[str]:
        """Column names the FrameBuilder assembles the embedding into."""
        return [f"cls_{i}" for i in range(self.DIM)]
