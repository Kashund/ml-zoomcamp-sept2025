from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class TabularModelConfig:
    cat_sizes: List[int]  # vocab size per categorical col (including 0=UNK)
    num_dim: int          # numeric + multihot dim
    emb_dim: int = 16
    hidden: int = 128
    dropout: float = 0.2


class HitNet(nn.Module):
    """Small tabular model: embeddings for categoricals + MLP for concat."""

    def __init__(self, cfg: TabularModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embs = nn.ModuleList(
            [nn.Embedding(size, cfg.emb_dim) for size in cfg.cat_sizes]
        )

        in_dim = cfg.emb_dim * len(cfg.cat_sizes) + cfg.num_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden // 2, 1),
        )

    def forward(self, cats: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        # cats: [B, n_cat] int64, nums: [B, num_dim] float32
        embs = []
        for j, emb in enumerate(self.embs):
            embs.append(emb(cats[:, j]))
        x = torch.cat(embs + [nums], dim=1)
        logits = self.net(x)
        return logits.squeeze(1)
