import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from .preprocess import load_details, build_preproc, make_tabular_matrix, make_hit_label
from .model_torch import HitNet, TabularModelConfig


class TabDataset(Dataset):
    def __init__(self, cats: np.ndarray, nums: np.ndarray, y: np.ndarray):
        self.cats = cats
        self.nums = nums
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.cats[idx]).long(),
            torch.from_numpy(self.nums[idx]).float(),
            torch.tensor(self.y[idx]).float(),
        )


def train_one_epoch(model, loader, opt, loss_fn, device: str) -> float:
    model.train()
    total = 0.0
    n = 0
    for cats, nums, y in loader:
        cats, nums, y = cats.to(device), nums.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(cats, nums)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total += float(loss.item()) * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device: str, threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    logits_all = []
    y_all = []
    for cats, nums, y in loader:
        cats, nums = cats.to(device), nums.to(device)
        logits = model(cats, nums).detach().cpu().numpy()
        logits_all.append(logits)
        y_all.append(y.numpy())
    logits = np.concatenate(logits_all)
    y = np.concatenate(y_all).astype(int)

    prob = 1.0 / (1.0 + np.exp(-logits))

    out: Dict[str, float] = {}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, prob))
        out["pr_auc"] = float(average_precision_score(y, prob))
    else:
        # edge case: validation split ended up with a single class
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")

    pred = (prob >= threshold).astype(int)
    out["f1"] = float(f1_score(y, pred, zero_division=0))

    pos = float(prob[y == 1].mean()) if (y == 1).any() else float("nan")
    neg = float(prob[y == 0].mean()) if (y == 0).any() else float("nan")
    out["pos_mean"] = pos
    out["neg_mean"] = neg
    out["sep"] = pos - neg if np.isfinite(pos) and np.isfinite(neg) else float("nan")
    return out


@torch.no_grad()
def _collect_probs(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    y_all = []
    for cats, nums, y in loader:
        cats, nums = cats.to(device), nums.to(device)
        logits = model(cats, nums).detach().cpu().numpy()
        logits_all.append(logits)
        y_all.append(y.numpy())
    logits = np.concatenate(logits_all)
    y_true = np.concatenate(y_all).astype(int)
    y_prob = 1.0 / (1.0 + np.exp(-logits))
    return y_true, y_prob


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _best_threshold_from_valid_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best_idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", default="data/details.csv")
    ap.add_argument("--topk", type=float, default=0.20, help="Top fraction of members treated as hit label.")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for reporting F1 during eval.")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = load_details(args.details)

    # label uses members only during training; do NOT include members as a feature
    y = make_hit_label(df, topk_fraction=args.topk).values.astype(np.int64)

    # keep only rows with usable members for label creation
    members_num = pd.to_numeric(df.get("members"), errors="coerce")
    mask = members_num.notna().values
    df = df.loc[mask].reset_index(drop=True)
    y = y[mask]

    # Multi-hot list columns (skip missing columns automatically)
    topk_map_full: Dict[str, int] = {
        "genres": 25,
        "themes": 25,
        "demographics": 10,
        "studios": 25,
    }
    topk_map = {k: v for k, v in topk_map_full.items() if k in df.columns}
    missing = [k for k in topk_map_full.keys() if k not in topk_map]
    if missing:
        print(f"warning: missing list columns in details.csv (skipping): {missing}")

    # 80/10/10 split (stratified)
    train_df, temp_df, y_train, y_temp = train_test_split(
        df,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )
    valid_df, test_df, y_valid, y_test = train_test_split(
        temp_df,
        y_temp,
        test_size=0.5,
        random_state=args.seed,
        stratify=y_temp,
    )

    # IMPORTANT: build preproc/vocabs on *train only* to avoid leakage
    cfg = build_preproc(train_df, topk=topk_map)
    cats_tr, nums_tr = make_tabular_matrix(train_df, cfg)
    cats_va, nums_va = make_tabular_matrix(valid_df, cfg)
    cats_te, nums_te = make_tabular_matrix(test_df, cfg)

    train_ds = TabDataset(cats_tr, nums_tr, y_train)
    val_ds = TabDataset(cats_va, nums_va, y_valid)
    test_ds = TabDataset(cats_te, nums_te, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    cat_sizes = [max(v.values()) + 1 for v in cfg.cat_vocab.values()]
    model_cfg = TabularModelConfig(cat_sizes=cat_sizes, num_dim=nums_tr.shape[1])

    device = _pick_device()
    model = HitNet(model_cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = -1e18
    best_metrics: Dict[str, float] = {}

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        m = eval_epoch(model, val_loader, device, threshold=args.threshold)

        primary = m["roc_auc"] if np.isfinite(m["roc_auc"]) else m.get("sep", float("-inf"))
        print(
            "epoch={epoch} train_loss={tr_loss:.4f} roc_auc={roc:.4f} pr_auc={pr:.4f} f1={f1:.4f}".format(
                epoch=epoch,
                tr_loss=tr_loss,
                roc=m["roc_auc"] if np.isfinite(m["roc_auc"]) else float("nan"),
                pr=m["pr_auc"] if np.isfinite(m["pr_auc"]) else float("nan"),
                f1=m["f1"],
            )
        )

        if primary > best:
            best = primary
            best_metrics = dict(m)

            torch.save(model.state_dict(), out_dir / "hitnet.pt")
            (out_dir / "preproc.json").write_text(json.dumps(cfg.to_dict(), indent=2))
            (out_dir / "hitnet_val_metrics.json").write_text(json.dumps(best_metrics, indent=2))
            print(
                f"saved: {out_dir/'hitnet.pt'} | {out_dir/'preproc.json'} | {out_dir/'hitnet_val_metrics.json'}"
            )

    # Reload best weights (safety) and pick threshold on validation to maximize F1
    model.load_state_dict(torch.load(out_dir / "hitnet.pt", map_location=device))
    yv_true, yv_prob = _collect_probs(model, val_loader, device)
    best_threshold = _best_threshold_from_valid_f1(yv_true, yv_prob)

    yt_true, yt_prob = _collect_probs(model, test_loader, device)
    test_pred = (yt_prob >= best_threshold).astype(int)

    hitnet_test = {
        "roc_auc": float(roc_auc_score(yt_true, yt_prob)) if len(np.unique(yt_true)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(yt_true, yt_prob)) if len(np.unique(yt_true)) == 2 else float("nan"),
        "f1": float(f1_score(yt_true, test_pred, zero_division=0)),
        "threshold": float(best_threshold),
        "seed": int(args.seed),
    }

    (out_dir / "hitnet_threshold.json").write_text(json.dumps({"threshold": best_threshold}, indent=2))
    (out_dir / "hitnet_metrics.json").write_text(json.dumps(hitnet_test, indent=2))

    print("done. best val metrics:", best_metrics)
    print("test metrics:", hitnet_test)


if __name__ == "__main__":
    main()
