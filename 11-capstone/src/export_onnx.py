import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .preprocess import PreprocConfig
from .model_torch import HitNet, TabularModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    art = Path(args.artifacts)
    preproc = PreprocConfig.from_dict(json.loads((art / "preproc.json").read_text()))

    cat_sizes = [max(v.values()) + 1 for v in preproc.cat_vocab.values()]
    # numeric dim = base nums + multihot
    num_dim = len(preproc.num_cols) + sum(len(v) for v in preproc.multilabel.values())

    model_cfg = TabularModelConfig(cat_sizes=cat_sizes, num_dim=num_dim)
    model = HitNet(model_cfg)
    model.load_state_dict(torch.load(art / "hitnet.pt", map_location="cpu"))
    model.eval()

    dummy_cats = torch.zeros((1, len(preproc.cat_cols)), dtype=torch.long)
    dummy_nums = torch.zeros((1, num_dim), dtype=torch.float32)

    onnx_path = art / "hitnet.onnx"
    torch.onnx.export(
        model,
        (dummy_cats, dummy_nums),
        onnx_path,
        input_names=["cats", "nums"],
        output_names=["logits"],
        dynamic_axes={"cats": {0: "batch"}, "nums": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print("saved:", onnx_path)


if __name__ == "__main__":
    main()
