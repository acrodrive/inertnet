"""Evaluation metrics: minADE_k / minFDE_k / MissRate@2m, plus occlusion slices.

Evaluated at the leaderboard-aligned anchor step t0 = current_time_index. The
model still runs its full per-step recurrence; we just read the trajectory head
output at the eligible step whose index == t0.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def trajectory_metrics(out: dict, batch: dict, cfg, miss_thresh: float = 2.0) -> dict:
    elig = out["elig_idx"]
    t0 = int(batch["current_time_index"][0].item())
    if t0 not in elig.tolist():
        # fall back to the last eligible step
        e = len(elig) - 1
        t0 = int(elig[e].item())
    else:
        e = elig.tolist().index(t0)

    traj = out["traj"][:, :, e]                       # [B,S,K,N,2]
    logits = out["mode_logits"][:, :, e]              # [B,S,K]
    B, S, T = batch["gt_valid"].shape
    N = cfg.horizon
    off = torch.arange(1, N + 1, device=traj.device)
    steps = (t0 + off).clamp(max=T - 1)
    in_range = (t0 + off) < T
    fut = batch["gt_pos"][:, :, :, :2][:, :, steps, :]      # [B,S,N,2]
    fut_v = batch["gt_valid"][:, :, steps] & in_range[None, None]

    scale = cfg.pos_scale
    err = torch.linalg.vector_norm(traj - fut[:, :, None], dim=-1) * scale   # [B,S,K,N] metres
    fvf = fut_v[:, :, None].float()
    ade_k = (err * fvf).sum(-1) / fvf.sum(-1).clamp_min(1.0)         # [B,S,K]
    last = fut_v.float().cumsum(-1).argmax(-1)                       # [B,S] last valid idx
    fde_k = torch.gather(err, 3, last[:, :, None, None].expand(-1, -1, err.size(2), 1)).squeeze(-1)

    sel = batch["is_focal"] & batch["slot_mask"] & fut_v.any(-1)     # [B,S]

    def slice_metrics(extra_mask: torch.Tensor | None) -> dict:
        m = sel if extra_mask is None else (sel & extra_mask)
        if m.sum() == 0:
            return {"minADE": float("nan"), "minFDE": float("nan"),
                    "MissRate": float("nan"), "n": 0}
        min_ade = ade_k.min(-1).values[m]
        min_fde = fde_k.min(-1).values[m]
        miss = (min_fde > miss_thresh).float()
        return {
            "minADE": min_ade.mean().item(),
            "minFDE": min_fde.mean().item(),
            "MissRate": miss.mean().item(),
            "n": int(m.sum().item()),
        }

    # occlusion slice: focal agents that had ANY masked/occluded history step
    had_gap = (batch["synth_mask"] | (batch["recon_mask"] & ~batch["obj_valid"]))
    had_gap = had_gap[:, :, : t0 + 1].any(-1)

    return {
        "all": slice_metrics(None),
        "occluded": slice_metrics(had_gap),
        "clean": slice_metrics(~had_gap),
    }


@torch.no_grad()
def occlusion_reconstruction_error(out: dict, batch: dict, cfg) -> dict:
    """ADE on synthetically-masked steps: does f_t recover the hidden position?"""
    m = batch["synth_mask"] & batch["slot_mask"][:, :, None]
    if m.sum() == 0:
        return {"recon_ade_ft": float("nan"), "recon_ade_fupd": float("nan"), "n": 0}
    tgt = batch["gt_pos"]
    scale = cfg.pos_scale
    e_ft = torch.linalg.vector_norm(out["ft_box"]["pos"] - tgt, dim=-1) * scale
    e_fu = torch.linalg.vector_norm(out["fupd_box"]["pos"] - tgt, dim=-1) * scale
    return {
        "recon_ade_ft": e_ft[m].mean().item(),
        "recon_ade_fupd": e_fu[m].mean().item(),
        "n": int(m.sum().item()),
    }
