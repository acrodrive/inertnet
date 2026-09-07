"""Baseline A (spec 3.4.2): constant velocity / constant turn-rate extrapolation.

No learning. Reads the last valid state before t0 and integrates forward. Used as
the sanity floor for minADE/minFDE.

    python -m mambahybrid.baselines --config configs/default.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import WOMDDataset, collate


def constant_velocity(batch: dict, cfg: Config) -> dict:
    gt_pos = batch["gt_pos"][..., :2].numpy() * cfg.pos_scale     # [B,S,T,2] metres
    gt_valid = batch["gt_valid"].numpy()
    B, S, T, _ = gt_pos.shape
    t0 = int(batch["current_time_index"][0])
    N = cfg.horizon
    dt = 0.1

    ade, fde = [], []
    focal = (batch["is_focal"] & batch["slot_mask"]).numpy()
    for b in range(B):
        for s in range(S):
            if not focal[b, s]:
                continue
            hist = np.where(gt_valid[b, s, : t0 + 1])[0]
            fut = np.where(gt_valid[b, s, t0 + 1 : t0 + 1 + N])[0]
            if len(hist) < 2 or len(fut) == 0:
                continue
            p1 = gt_pos[b, s, hist[-1]]
            p0 = gt_pos[b, s, hist[-2]]
            v = (p1 - p0) / (dt * (hist[-1] - hist[-2]))
            steps = np.arange(1, N + 1)[:, None]
            pred = p1[None] + v[None] * (steps * dt)              # [N,2]
            tgt = gt_pos[b, s, t0 + 1 : t0 + 1 + N]
            err = np.linalg.norm(pred - tgt, axis=1)[fut]
            ade.append(err.mean())
            fde.append(err[-1])
    return {"minADE": float(np.mean(ade)), "minFDE": float(np.mean(fde)),
            "MissRate": float(np.mean(np.array(fde) > 2.0)), "n": len(ade)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--shards", default=None)
    ap.add_argument("--max-batches", type=int, default=200)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    ds = WOMDDataset(args.shards or cfg.val_shards, cfg.preprocess(), "val",
                     cfg.cache_dir, cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, collate_fn=collate)

    tot = {"minADE": [], "minFDE": [], "MissRate": []}
    for i, batch in enumerate(loader):
        if i >= args.max_batches:
            break
        m = constant_velocity(batch, cfg)
        for k in tot:
            tot[k].append(m[k])
    print({k: round(float(np.mean(v)), 3) for k, v in tot.items()})


if __name__ == "__main__":
    main()
