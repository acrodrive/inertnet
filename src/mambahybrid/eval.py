"""Standalone evaluation + ablation runner.

    python -m mambahybrid.eval --config configs/default.yaml --ckpt runs/exp1/best.pt
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import WOMDDataset, collate
from .model import MambaHybrid
from .train import evaluate, move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", default=None, help="override val_shards glob")
    ap.add_argument("--max-batches", type=int, default=500)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"

    ds = WOMDDataset(args.shards or cfg.val_shards, cfg.preprocess(), "val",
                     cfg.cache_dir, cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate)

    model = MambaHybrid(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.ckpt} (step {ckpt.get('step', '?')})")

    metrics = evaluate(model, loader, cfg, device, max_batches=args.max_batches)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
