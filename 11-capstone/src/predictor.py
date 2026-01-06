"""Unified prediction layer for both backends.

Default backend: Random Forest (scikit-learn) trained by ``src/train_rf.py``.
Optional backend: HitNet (PyTorch -> ONNX) as a bonus workflow.

Environment variables:
  MODEL_BACKEND: "rf" (default) or "hitnet"

RF artifacts (defaults under artifacts/):
  RF_MODEL_PATH:      artifacts/rf_pipeline.joblib
  RF_META_PATH:       artifacts/rf_meta.json
  RF_THRESHOLD_PATH:  artifacts/rf_threshold.json

HitNet artifacts (defaults under artifacts/):
  ONNX_PATH:          artifacts/hitnet.onnx
  PREPROC_PATH:       artifacts/preproc.json
  THRESHOLD:          probability cutoff (if no *_threshold.json exists)
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import joblib
import numpy as np
import pandas as pd

from .preprocess import parse_listlike
from .rf_features import apply_vocab_map, ensure_columns


def _default_artifacts_dir() -> Path:
    """Find artifacts/ relative to common execution locations."""
    candidates = [
        Path.cwd() / "artifacts",
        Path(__file__).resolve().parent / "artifacts",  # if running from repo root
        Path(__file__).resolve().parent.parent / "artifacts",  # if running from src/
        Path("/var/task") / "artifacts",  # Lambda container
        Path("/app") / "artifacts",  # Docker
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _env_path(name: str, default: Path) -> str:
    return os.getenv(name, str(default))


def _load_threshold(path: str, fallback_env: str = "THRESHOLD", fallback: float = 0.5) -> float:
    try:
        p = Path(path)
        if p.exists():
            d = json.loads(p.read_text())
            if isinstance(d, dict) and "threshold" in d:
                return float(d["threshold"])
    except Exception:
        pass
    return float(os.getenv(fallback_env, str(fallback)))


def _normalize_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Ensure list fields are lists (not strings)."""
    out = dict(record)
    for k in ["genres", "themes", "demographics", "studios"]:
        v = out.get(k, [])
        if v is None:
            out[k] = []
        elif isinstance(v, str):
            out[k] = parse_listlike(v)
        else:
            out[k] = list(v)
    return out


@lru_cache(maxsize=1)
def _load_rf_bundle() -> Tuple[Any, Dict[str, Any], float]:
    art_dir = _default_artifacts_dir()
    model_path = Path(_env_path("RF_MODEL_PATH", art_dir / "rf_pipeline.joblib"))
    meta_path = Path(_env_path("RF_META_PATH", art_dir / "rf_meta.json"))
    thr_path = _env_path("RF_THRESHOLD_PATH", art_dir / "rf_threshold.json")

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"RF artifacts not found. Expected {model_path} and {meta_path}. "
            "Run: make train (or python -m src.train_rf) to generate them."
        )

    pipe = joblib.load(model_path)
    meta = json.loads(meta_path.read_text())
    threshold = _load_threshold(thr_path)
    return pipe, meta, threshold


def _rf_features_from_record(record: Mapping[str, Any], meta: Mapping[str, Any]) -> pd.DataFrame:
    r = _normalize_record(record)
    cat_cols = list(meta.get("cat_cols", []))
    num_cols = list(meta.get("num_cols", []))
    bin_cols = list(meta.get("bin_cols", []))
    vocab_map = meta.get("vocab_map", {})

    # Build a 1-row DataFrame with base columns + multi cols
    base: Dict[str, Any] = {}
    for c in cat_cols:
        base[c] = r.get(c, "__MISSING__") if r.get(c, None) is not None else "__MISSING__"
    for c in num_cols:
        v = r.get(c, None)
        try:
            base[c] = float(v) if v is not None else np.nan
        except Exception:
            base[c] = np.nan
    for c in vocab_map.keys():
        base[c] = r.get(c, [])

    df = pd.DataFrame([base])
    df = apply_vocab_map(df, vocab_map=vocab_map, drop_original=True)

    # Ensure expected columns exist
    df = ensure_columns(df, cat_cols + num_cols + bin_cols, fill_value=0)
    df = df[cat_cols + num_cols + bin_cols]
    return df


def predict_proba(record: Mapping[str, Any]) -> Tuple[float, float, str]:
    """Return (proba, threshold, backend)."""
    backend = os.getenv("MODEL_BACKEND", "rf").strip().lower()
    art_dir = _default_artifacts_dir()

    if backend == "hitnet":
        onnx_path = _env_path("ONNX_PATH", art_dir / "hitnet.onnx")
        preproc_path = _env_path("PREPROC_PATH", art_dir / "preproc.json")
        threshold = _load_threshold(_env_path("HITNET_THRESHOLD_PATH", art_dir / "hitnet_threshold.json"))
        try:
            from .inference_onnx import predict_proba_onnx  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "HitNet backend requires onnxruntime. Install requirements or switch to MODEL_BACKEND=rf."
            ) from e

        proba = float(predict_proba_onnx(dict(record), onnx_path, preproc_path))
        return proba, threshold, "hitnet"

    # default: RF
    pipe, meta, threshold = _load_rf_bundle()
    X = _rf_features_from_record(record, meta)
    proba = float(pipe.predict_proba(X)[:, 1][0])
    return proba, threshold, "rf"


def predict(record: Mapping[str, Any]) -> Dict[str, Any]:
    proba, threshold, backend = predict_proba(record)
    return {
        "hit_probability": float(proba),
        "hit": bool(proba >= threshold),
        "threshold": float(threshold),
        "backend": backend,
    }
