import json
from typing import Any, Dict, Tuple

import numpy as np
import onnxruntime as ort

from .preprocess import PreprocConfig, parse_listlike


def preprocess_record(record: Dict[str, Any], cfg: PreprocConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Single-record preprocess (cats, nums) matching the ONNX model."""
    # normalize list columns
    for col in cfg.multilabel.keys():
        v = record.get(col, [])
        if isinstance(v, str):
            v = parse_listlike(v)
        record[col] = v if v is not None else []

    # cats
    cats = np.zeros((1, len(cfg.cat_cols)), dtype=np.int64)
    for j, c in enumerate(cfg.cat_cols):
        v = record.get(c, "__MISSING__")
        if v is None:
            v = "__MISSING__"
        v = str(v)
        cats[0, j] = cfg.cat_vocab[c].get(v, 0)

    # numeric base
    nums = []
    for c in cfg.num_cols:
        val = record.get(c, None)
        try:
            val = float(val)
        except Exception:
            val = np.nan
        nums.append(val)
    # fill NaNs with 0 (for serverless simplicity); training uses medians
    nums = np.array(nums, dtype=np.float32)
    nums = np.nan_to_num(nums, nan=0.0)

    # multihot
    mh = []
    for col, labels in cfg.multilabel.items():
        lst = record.get(col, [])
        for lab in labels:
            mh.append(1.0 if lab in lst else 0.0)
    mh = np.array(mh, dtype=np.float32)

    full = np.concatenate([nums, mh], axis=0).reshape(1, -1).astype(np.float32)
    return cats, full


def predict_proba_onnx(record: Dict[str, Any], onnx_path: str, preproc_path: str) -> float:
    cfg = PreprocConfig.from_dict(json.loads(open(preproc_path, "r").read()))
    cats, nums = preprocess_record(record, cfg)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    logits = sess.run(None, {"cats": cats, "nums": nums})[0].reshape(-1)
    prob = 1.0 / (1.0 + np.exp(-logits))[0]
    return float(prob)
