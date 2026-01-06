"""Random-Forest feature helpers.

This module is shared by:
- src/train_rf.py (training-time feature building)
- src/predictor.py (runtime feature building)

It mirrors the notebook's baseline approach:
1) Keep base categorical + numeric columns
2) Expand selected multi-label columns (e.g., genres) into fixed multi-hot columns
   using a top-k vocabulary learned on the training split.

We intentionally keep column names identical to the notebook: ``{col}__{token}``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .preprocess import parse_listlike


def build_topk_vocab(values: Iterable[Any], k: int) -> List[str]:
    """Build a top-k token vocabulary from a column of list-ish values."""
    c: Counter = Counter()
    for v in values:
        if v is None:
            continue
        if isinstance(v, str):
            tokens = parse_listlike(v)
        elif isinstance(v, (list, tuple, set)):
            tokens = [str(x).strip() for x in v if str(x).strip()]
        else:
            tokens = parse_listlike(str(v))
        c.update(tokens)
    return [t for t, _ in c.most_common(k)]


def build_vocab_map(df: pd.DataFrame, multi_cols: Sequence[str], topk: int) -> Dict[str, List[str]]:
    """Return {multi_col: [topk tokens]} learned on the given dataframe."""
    vocab_map: Dict[str, List[str]] = {}
    for col in multi_cols:
        if col not in df.columns:
            continue
        vocab_map[col] = build_topk_vocab(df[col].values, k=topk)
    return vocab_map


def _to_token_set(v: Any) -> set:
    if v is None:
        return set()
    if isinstance(v, str):
        return set(parse_listlike(v))
    if isinstance(v, (list, tuple, set)):
        return set(str(x).strip() for x in v if str(x).strip())
    return set(parse_listlike(str(v)))


def add_multi_hot(df_in: pd.DataFrame, col: str, vocab: Sequence[str]) -> pd.DataFrame:
    """Add multi-hot columns for a single multi-label column."""
    df_out = df_in.copy()
    token_sets = df_out[col].apply(_to_token_set)
    for t in vocab:
        df_out[f"{col}__{t}"] = token_sets.apply(lambda s: 1 if t in s else 0).astype(np.int8)
    return df_out


def apply_vocab_map(
    df_in: pd.DataFrame,
    vocab_map: Mapping[str, Sequence[str]],
    drop_original: bool = True,
) -> pd.DataFrame:
    """Apply vocab_map to expand multi-label columns into multi-hot columns."""
    df = df_in.copy()
    for col, vocab in vocab_map.items():
        if col not in df.columns:
            # still create empty columns for stability
            for t in vocab:
                df[f"{col}__{t}"] = 0
            continue
        df = add_multi_hot(df, col, vocab)
    if drop_original:
        df = df.drop(columns=list(vocab_map.keys()), errors="ignore")
    return df


def ensure_columns(df_in: pd.DataFrame, columns: Sequence[str], fill_value: Any = 0) -> pd.DataFrame:
    """Ensure df has all columns; add missing ones with fill_value."""
    df = df_in.copy()
    for c in columns:
        if c not in df.columns:
            df[c] = fill_value
    return df
