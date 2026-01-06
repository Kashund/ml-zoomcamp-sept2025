"""Train the production Random Forest baseline.

This script is the source of truth for the repo's default deployment model.

Workflow:
1) Build a cold-start label from `members` (top fraction)
2) Create an 80/10/10 train/valid/test split (stratified)
3) Expand multi-label columns into fixed multi-hot columns using a top-k vocab
4) Fit a scikit-learn pipeline (impute+OHE+scaler + RandomForest)
5) Choose threshold on validation to maximize F1
6) Report test ROC-AUC / PR-AUC / F1 and write artifacts under artifacts/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocess import load_details, make_hit_label
from .rf_features import apply_vocab_map, build_vocab_map, ensure_columns


BASE_CATS_DEFAULT = ["type", "season", "source", "rating", "status"]
BASE_NUMS_DEFAULT = ["year", "episodes"]
MULTI_COLS_DEFAULT = ["genres", "themes", "demographics", "studios"]
TARGET_COL = "is_hit"


def best_threshold_from_valid_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    # thresholds has length n-1; align using f1[:-1]
    best_idx = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best_idx])


def build_pipeline(final_cat: List[str], final_num: List[str], final_bin: List[str], n_estimators: int, seed: int):
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    bin_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0))])

    preprocess = ColumnTransformer(
        transformers=[
            ("cats", cat_pipe, final_cat),
            ("nums", num_pipe, final_num),
            ("bins", bin_pipe, final_bin),
        ],
        remainder="drop",
    )

    est = RandomForestClassifier(
        n_estimators=int(n_estimators),
        random_state=int(seed),
        n_jobs=-1,
    )

    return Pipeline(steps=[("prep", preprocess), ("clf", est)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=str, default="data/details.csv")
    parser.add_argument("--artifacts", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk_fraction", type=float, default=0.20)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--n_estimators", type=int, default=400)
    args = parser.parse_args()

    details_path = Path(args.details)
    art_dir = Path(args.artifacts)
    art_dir.mkdir(parents=True, exist_ok=True)

    # Load and parse list-like columns
    df = load_details(str(details_path))

    # Label (allowed leakage source for training ONLY)
    if "members" not in df.columns:
        raise ValueError("details.csv must contain 'members' to build the hit label")
    df[TARGET_COL] = make_hit_label(df, topk_fraction=float(args.topk_fraction))

    base_cats = [c for c in BASE_CATS_DEFAULT if c in df.columns]
    base_nums = [c for c in BASE_NUMS_DEFAULT if c in df.columns]
    multi_cols = [c for c in MULTI_COLS_DEFAULT if c in df.columns]

    # Build canonical 3-way split once (80/10/10)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=int(args.seed),
        stratify=df[TARGET_COL],
    )
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=int(args.seed),
        stratify=temp_df[TARGET_COL],
    )

    # Build vocab on TRAIN ONLY
    vocab_map = build_vocab_map(train_df, multi_cols=multi_cols, topk=int(args.topk))

    def make_X(df_in: pd.DataFrame) -> pd.DataFrame:
        X = df_in[base_cats + base_nums + list(vocab_map.keys())].copy()
        X = apply_vocab_map(X, vocab_map=vocab_map, drop_original=True)
        return X

    X_train = make_X(train_df)
    y_train = train_df[TARGET_COL].to_numpy()
    X_valid = make_X(valid_df)
    y_valid = valid_df[TARGET_COL].to_numpy()
    X_test = make_X(test_df)
    y_test = test_df[TARGET_COL].to_numpy()

    final_cat = [c for c in base_cats if c in X_train.columns]
    final_num = [c for c in base_nums if c in X_train.columns]
    final_bin = [c for c in X_train.columns if "__" in c]

    # Ensure stability across splits
    X_valid = ensure_columns(X_valid, final_cat + final_num + final_bin, fill_value=0)
    X_test = ensure_columns(X_test, final_cat + final_num + final_bin, fill_value=0)
    X_valid = X_valid[final_cat + final_num + final_bin]
    X_test = X_test[final_cat + final_num + final_bin]
    X_train = X_train[final_cat + final_num + final_bin]

    pipe = build_pipeline(final_cat, final_num, final_bin, n_estimators=args.n_estimators, seed=args.seed)
    pipe.fit(X_train, y_train)

    p_valid = pipe.predict_proba(X_valid)[:, 1]
    t = best_threshold_from_valid_f1(y_valid, p_valid)

    p_test = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else float("nan"),
        "f1": float(f1_score(y_test, (p_test >= t).astype(int), zero_division=0)),
        "threshold": float(t),
        "seed": int(args.seed),
        "topk_fraction": float(args.topk_fraction),
        "topk": int(args.topk),
        "n_estimators": int(args.n_estimators),
        "hit_rate_train": float(np.mean(y_train)),
        "hit_rate_valid": float(np.mean(y_valid)),
        "hit_rate_test": float(np.mean(y_test)),
    }

    # ---- Write artifacts ----
    model_path = art_dir / "rf_pipeline.joblib"
    meta_path = art_dir / "rf_meta.json"
    metrics_path = art_dir / "rf_metrics.json"
    thresh_path = art_dir / "rf_threshold.json"

    joblib.dump(pipe, model_path)

    meta = {
        "model": "rf",
        "strategy": "impute",
        "seed": int(args.seed),
        "topk_fraction": float(args.topk_fraction),
        "topk": int(args.topk),
        "n_estimators": int(args.n_estimators),
        "target_col": TARGET_COL,
        "cat_cols": final_cat,
        "num_cols": final_num,
        "bin_cols": final_bin,
        "multi_cols": list(vocab_map.keys()),
        "vocab_map": vocab_map,
        "splits": {"train": 0.8, "valid": 0.1, "test": 0.1},
    }

    meta_path.write_text(json.dumps(meta, indent=2))
    metrics_path.write_text(json.dumps(metrics, indent=2))
    thresh_path.write_text(json.dumps({"threshold": float(t)}, indent=2))

    print("Saved:")
    print(" -", model_path)
    print(" -", meta_path)
    print(" -", metrics_path)
    print(" -", thresh_path)
    print("Test metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
