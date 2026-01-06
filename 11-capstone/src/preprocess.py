import re
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


LIST_COLS_DEFAULT = [
    "genres",
    "themes",
    "demographics",
    "studios",
    "producers",
    "licensors",
    "streaming",
    "explicit_genres",
]

LEAKAGE_COLS = {
    # engagement / post-release or platform-derived aggregates
    "members",
    "favorites",
    "scored_by",
    "score",
    "rank",
    "popularity",
    # dates can be unknown at announcement; keep only if you explicitly allow them
    "start_date",
    "end_date",
}


def _is_nullish(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip().lower() in {"", "nan", "none", "null"}:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    try:
        if isinstance(x, np.floating) and np.isnan(x):
            return True
    except Exception:
        pass
    # optional pandas NA handling (training env)
    try:
        import pandas as pd  # type: ignore
        return bool(pd.isna(x))
    except Exception:
        return False


def parse_listlike(x: Any) -> List[str]:
    """Parse list-ish string fields into a list of tokens."""
    if _is_nullish(x):
        return []
    s = str(x).strip().replace("'", "").replace('"', "")
    for ch in "[]{}":
        s = s.replace(ch, "")
    return [p.strip() for p in re.split(r"[;,]", s) if p.strip()]


def topk_labels(values: Iterable[List[str]], k: int) -> List[str]:
    counts: Dict[str, int] = {}
    for lst in values:
        for it in lst:
            counts[it] = counts.get(it, 0) + 1
    return [lab for lab, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:k]]


@dataclass
class PreprocConfig:
    cat_cols: List[str]
    num_cols: List[str]
    multilabel: Dict[str, List[str]]  # column -> allowed labels (top-k)
    cat_vocab: Dict[str, Dict[str, int]]  # col -> value -> index (0 reserved for unknown)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "multilabel": self.multilabel,
            "cat_vocab": self.cat_vocab,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PreprocConfig":
        return PreprocConfig(
            cat_cols=list(d["cat_cols"]),
            num_cols=list(d["num_cols"]),
            multilabel=dict(d["multilabel"]),
            cat_vocab=dict(d["cat_vocab"]),
        )


def build_preproc(
    df_details,
    topk: Dict[str, int],
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
) -> PreprocConfig:
    """Fit preprocessing config from the training dataframe."""
    if cat_cols is None:
        cat_cols = ["type", "season", "source", "rating", "status"]
    if num_cols is None:
        num_cols = ["year", "episodes"]

    multilabel: Dict[str, List[str]] = {}
    for col, k in topk.items():
        multilabel[col] = topk_labels(df_details[col], k)

    cat_vocab: Dict[str, Dict[str, int]] = {}
    for c in cat_cols:
        values = df_details[c].fillna("__MISSING__").astype(str).value_counts().index.tolist()
        # 0 = unknown
        cat_vocab[c] = {v: i + 1 for i, v in enumerate(values)}

    return PreprocConfig(cat_cols=cat_cols, num_cols=num_cols, multilabel=multilabel, cat_vocab=cat_vocab)


def make_tabular_matrix(df, cfg: PreprocConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert df into (cats, nums) where:
    - cats: int64 [n, n_cat]
    - nums: float32 [n, n_num + n_multihot]
    """
    import pandas as pd  # training-time dependency

    cats = np.zeros((len(df), len(cfg.cat_cols)), dtype=np.int64)
    for j, c in enumerate(cfg.cat_cols):
        col = df[c].fillna("__MISSING__").astype(str).values
        vocab = cfg.cat_vocab[c]
        cats[:, j] = np.array([vocab.get(v, 0) for v in col], dtype=np.int64)

    num_df = df[cfg.num_cols].copy()
    for c in cfg.num_cols:
        num_df[c] = pd.to_numeric(num_df[c], errors="coerce")
    num_df = num_df.fillna(num_df.median(numeric_only=True))
    nums = num_df.values.astype(np.float32)

    multihot_data = []
    for col, labels in cfg.multilabel.items():
        for lab in labels:
            multihot_data.append(df[col].apply(lambda lst: int(lab in lst)).values.astype(np.float32))
    if multihot_data:
        mh = np.vstack(multihot_data).T
        nums = np.hstack([nums, mh])

    return cats, nums


def load_details(path: str, list_cols: Optional[List[str]] = None):
    import pandas as pd  # training-time dependency

    df = pd.read_csv(path)
    if list_cols is None:
        list_cols = [c for c in LIST_COLS_DEFAULT if c in df.columns]
    for c in list_cols:
        df[c] = df[c].apply(parse_listlike)
    return df


def make_hit_label(df_details, topk_fraction: float = 0.10):
    import pandas as pd  # training-time dependency

    members = pd.to_numeric(df_details["members"], errors="coerce")
    thr = members.quantile(1 - topk_fraction)
    return (members >= thr).astype(int)
