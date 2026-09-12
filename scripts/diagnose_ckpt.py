"""Diagnose a trained checkpoint: is the trajectory head multi-modal, and does
it beat a constant-velocity baseline?

exp1/exp2/exp3 all plateaued well short of a usable minADE. This tells us WHY
before we design the next change:

  * mode collapse  -> minADE_1 (top prob) == minADE_K (best of K), tiny spread
                      between the K predicted endpoints, one mode wins every time
  * mean regression -> predicted paths much shorter / straighter than GT
  * no better than trivial -> model minADE >= constant-velocity minADE
  * where it breaks -> ADE by horizon (1s..5s) and clean vs occluded

    python scripts/diagnose_ckpt.py --ckpt runs/exp3/best.pt [--max-batches 80]

Reads the anchor step t0 = current_time_index, focal agents only (the metric
population). Prints a report; writes nothing.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from mambahybrid.config import Config
from mambahybrid.data import WOMDDataset, collate
from mambahybrid.metrics import occlusion_gap_slices
from mambahybrid.model import MambaHybrid
from mambahybrid.train import evaluate, move


def _anchor_slice(out, batch, cfg):
    """Return traj [n,K,N,2] (metres), logits [n,K], gt [n,N,2] (m), gt_valid
    [n,N], p0 [n,2] (m), v0 [n,2] (m/s) — all for focal agents at t0."""
    elig = out["elig_idx"].tolist()
    t0 = int(batch["current_time_index"][0].item())
    e = elig.index(t0) if t0 in elig else len(elig) - 1
    t0 = elig[e]

    B, S, T = batch["gt_valid"].shape
    N, scale = cfg.horizon, cfg.pos_scale
    dev = batch["gt_valid"].device
    off = torch.arange(1, N + 1, device=dev)
    steps = (t0 + off).clamp(max=T - 1)
    in_rng = (t0 + off) < T

    traj = out["traj"][:, :, e] * scale                       # [B,S,K,N,2] m
    logits = out["mode_logits"][:, :, e]                      # [B,S,K]
    gt = batch["gt_pos"][:, :, steps, :2] * scale             # [B,S,N,2] m
    gv = batch["gt_valid"][:, :, steps] & in_rng[None, None]
    p0 = batch["gt_pos"][:, :, t0, :2] * scale                # [B,S,2] m
    v0 = batch["gt_vel"][:, :, t0] * cfg.vel_scale            # [B,S,2] m/s

    sel = batch["is_focal"] & batch["slot_mask"] & gv.any(-1)  # [B,S]
    gaps = occlusion_gap_slices(batch, t0)
    occ, nat = gaps["any"], gaps["natural"]

    return (traj[sel], logits[sel], gt[sel], gv[sel], p0[sel], v0[sel], occ[sel], nat[sel])


def _ade_fde(pred, gt, gv):
    """pred [n,(K),N,2], gt [n,N,2], gv [n,N] -> ade [n,(K)], fde [n,(K)]."""
    multi = pred.dim() == 4
    g = gt[:, None] if multi else gt
    v = gv[:, None] if multi else gv
    err = torch.linalg.vector_norm(pred - g, dim=-1)          # [n,(K),N]
    vf = v.float()
    ade = (err * vf).sum(-1) / vf.sum(-1).clamp_min(1.0)
    last = gv.float().cumsum(-1).argmax(-1)                   # [n]
    li = last[:, None, None].expand(*err.shape[:-1], 1) if multi else last[:, None]
    fde = torch.gather(err, -1, li).squeeze(-1)
    return ade, fde


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--shards", default=None)
    ap.add_argument("--max-batches", type=int, default=80)
    ap.add_argument("--device", default=None,
                    help="force cpu to avoid disturbing a training run on the GPU")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cfg.compile = False
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = WOMDDataset(args.shards or cfg.val_shards, cfg.preprocess(), "val",
                     cfg.cache_dir, cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=max(1, cfg.num_workers // 2), collate_fn=collate)

    model = MambaHybrid(cfg).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {args.ckpt}  (step {ck.get('step', '?')})  K={cfg.n_modes} "
          f"horizon={cfg.horizon} ({cfg.horizon/10:.1f}s)\n")

    std = evaluate(model, loader, cfg, device, max_batches=args.max_batches)
    print("standard metrics (train.evaluate):")
    for k in ("all/minADE", "all/minFDE", "all/MissRate",
              "occluded/minADE", "occluded/MissRate", "clean/minADE",
              "synth_occluded/minADE", "natural_occluded/minADE"):
        if k in std:
            print(f"  {k:22s} {std[k]:.3f}")

    T = tr = lg = None
    traj_a, logit_a, gt_a, gv_a, p0_a, v0_a, occ_a, nat_a = ([] for _ in range(8))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.max_batches:
                break
            batch = move(batch, device)
            out = model(batch)
            t, l, g, gv, p0, v0, occ, nat = _anchor_slice(out, batch, cfg)
            traj_a.append(t); logit_a.append(l); gt_a.append(g); gv_a.append(gv)
            p0_a.append(p0); v0_a.append(v0); occ_a.append(occ); nat_a.append(nat)
    traj = torch.cat(traj_a); logit = torch.cat(logit_a); gt = torch.cat(gt_a)
    gv = torch.cat(gv_a); p0 = torch.cat(p0_a); v0 = torch.cat(v0_a)
    occ = torch.cat(occ_a); nat = torch.cat(nat_a)
    n = traj.shape[0]
    K, N = cfg.n_modes, cfg.horizon
    print(f"\nfocal agents @ t0: {n}   (occluded {int(occ.sum())} "
          f"[synthetic {int((occ & ~nat).sum())} / natural {int(nat.sum())}], "
          f"clean {int((~occ).sum())})")

    # ---- multimodality ----
    ade_k, fde_k = _ade_fde(traj, gt, gv)                     # [n,K]
    best_k = ade_k.argmin(-1)
    top1 = logit.argmax(-1)
    ade_best = ade_k.gather(1, best_k[:, None]).squeeze(1)
    ade_top1 = ade_k.gather(1, top1[:, None]).squeeze(1)
    fde_best = fde_k.gather(1, best_k[:, None]).squeeze(1)

    print("\n-- multimodality --")
    print(f"  minADE_1 (top-prob mode)   {ade_top1.mean():.3f}")
    print(f"  minADE_K (best of {K})       {ade_best.mean():.3f}")
    print(f"  gain from K modes          {(ade_top1.mean() - ade_best.mean()):.3f}"
          f"   ({100*(1 - ade_best.mean()/ade_top1.mean()):.0f}% lower)")
    ends = traj[:, :, -1, :]                                  # [n,K,2] endpoints
    pair = torch.cdist(ends, ends)                            # [n,K,K]
    spread = pair.sum((-1, -2)) / (K * (K - 1))
    print(f"  mean pairwise endpoint spread   {spread.mean():.3f} m"
          f"   (collapsed if ~0; GT motion ~{torch.linalg.vector_norm(gt[:, -1] - p0, dim=-1).mean():.1f} m)")
    win = torch.bincount(best_k, minlength=K).float() / n
    prob = logit.softmax(-1).mean(0)
    print(f"  WTA winner share per mode   {[f'{x:.2f}' for x in win.tolist()]}")
    print(f"  mean softmax prob per mode  {[f'{x:.2f}' for x in prob.tolist()]}")
    eff = torch.exp(-(prob * prob.clamp_min(1e-9).log()).sum())
    print(f"  effective # modes (exp entropy of mean prob)   {eff:.2f} / {K}")

    # ---- constant-velocity baseline ----
    dt = torch.arange(1, N + 1, device=device).float() * 0.1
    cv = p0[:, None] + v0[:, None] * dt[None, :, None]        # [n,N,2] m
    cv_ade, cv_fde = _ade_fde(cv, gt, gv)
    print("\n-- constant-velocity baseline (single mode) --")
    print(f"  CV  minADE {cv_ade.mean():.3f}   minFDE {cv_fde.mean():.3f}")
    print(f"  model minADE_K {ade_best.mean():.3f}   minADE_1 {ade_top1.mean():.3f}")
    verdict = "BEATS CV" if ade_best.mean() < cv_ade.mean() else "WORSE THAN CV"
    print(f"  -> {verdict}")

    # ---- error vs horizon (best mode) ----
    print("\n-- ADE by horizon (best-of-K mode) --")
    bt = traj.gather(1, best_k[:, None, None, None].expand(-1, 1, N, 2)).squeeze(1)
    e_t = torch.linalg.vector_norm(bt - gt, dim=-1)           # [n,N]
    for s in range(9, N, 10):
        m = gv[:, s]
        if m.any():
            print(f"  t+{(s+1)/10:.1f}s   {e_t[m, s].mean():.3f} m")

    # ---- motion magnitude (mean regression check) ----
    gtd = torch.linalg.vector_norm(gt[:, -1] - p0, dim=-1)
    pmd = torch.linalg.vector_norm(bt[:, -1] - p0, dim=-1)
    cvd = torch.linalg.vector_norm(cv[:, -1] - p0, dim=-1)
    print("\n-- endpoint displacement from t0 (mean over focal) --")
    print(f"  GT {gtd.mean():.1f} m   model {pmd.mean():.1f} m   CV {cvd.mean():.1f} m"
          f"   (model << GT => regressing to a short/straight mean)")

    # ---- occluded vs clean (best-of-K), occluded split synthetic/natural ----
    for name, m in (("clean", ~occ), ("occluded", occ),
                    ("  synth", occ & ~nat), ("  natural", nat)):
        if m.any():
            print(f"\n  [{name:8s}] minADE_K {ade_best[m].mean():.3f}  "
                  f"minFDE_K {fde_best[m].mean():.3f}  n={int(m.sum())}")


if __name__ == "__main__":
    main()
