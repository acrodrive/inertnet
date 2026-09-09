"""Total loss (Specification 3.2.6).

    L_total = l_aux*L_aux + l_next*L_next
            + l_recon_ft*L_recon(f_t) + l_recon_fupd*L_recon(f_t_updated)
            + l_traj*L_traj_WTA + l_mode*L_mode_CE
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_BOX_W = {"pos": 1.0, "heading": 0.5, "size": 0.2, "vel": 0.5, "class": 0.2}


def _masked(loss_elem: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of loss over True entries of mask (mask broadcasts over trailing dims)."""
    while mask.dim() < loss_elem.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.to(loss_elem.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (loss_elem * mask).sum() / denom


def _recon_term(box: dict, batch: dict) -> torch.Tensor:
    m = batch["recon_mask"]                                   # [B,S,T]
    gt_head = torch.stack(
        [torch.sin(batch["gt_heading"]), torch.cos(batch["gt_heading"])], dim=-1
    )
    l_pos = _masked(F.smooth_l1_loss(box["pos"], batch["gt_pos"], reduction="none"), m)
    l_head = _masked(F.smooth_l1_loss(box["heading"], gt_head, reduction="none"), m)
    l_size = _masked(F.smooth_l1_loss(box["size"], batch["gt_size"], reduction="none"), m)
    l_vel = _masked(F.smooth_l1_loss(box["vel"], batch["gt_vel"], reduction="none"), m)
    T = m.shape[2]
    cls_t = batch["obj_class"][:, :, None].expand(-1, -1, T).reshape(-1)
    l_cls_elem = F.cross_entropy(
        box["class_logits"].reshape(-1, box["class_logits"].shape[-1]), cls_t,
        reduction="none",
    ).view_as(m)
    l_cls = _masked(l_cls_elem, m)
    return (
        _BOX_W["pos"] * l_pos + _BOX_W["heading"] * l_head + _BOX_W["size"] * l_size
        + _BOX_W["vel"] * l_vel + _BOX_W["class"] * l_cls
    )


def _gather_future(batch: dict, elig_idx: torch.Tensor, horizon: int):
    B, S, T = batch["gt_valid"].shape
    off = torch.arange(1, horizon + 1, device=elig_idx.device)
    steps = elig_idx[:, None] + off[None, :]                  # [E,N]
    in_range = steps < T
    idx = steps.clamp(max=T - 1)
    fut_pos = batch["gt_pos"][:, :, :, :2][:, :, idx, :]      # [B,S,E,N,2]
    fut_valid = batch["gt_valid"][:, :, idx] & in_range[None, None]
    return fut_pos, fut_valid


def compute_losses(out: dict, batch: dict, cfg) -> dict:
    recon_m = batch["recon_mask"]
    gv = batch["gt_valid"]

    # --- L_aux : prior predicts current position ---
    l_aux = _masked(
        F.smooth_l1_loss(out["prior_pos_now"], batch["gt_pos"], reduction="none"),
        recon_m,
    )

    # --- L_next : prior predicts next-step position ---
    l_next = _masked(
        F.smooth_l1_loss(
            out["prior_pos_next"][:, :, :-1], batch["gt_pos"][:, :, 1:], reduction="none"
        ),
        recon_m[:, :, :-1] & gv[:, :, 1:],
    )

    # --- L_recon on f_t and on f_t_updated ---
    l_recon_ft = _recon_term(out["ft_box"], batch)
    l_recon_fupd = _recon_term(out["fupd_box"], batch)

    # --- L_traj (WTA) + L_mode (CE) on focal agents ---
    traj = out["traj"]                                        # [B,S,E,K,N,2]
    fut_pos, fut_valid = _gather_future(batch, out["elig_idx"], cfg.horizon)
    focal = (batch["is_focal"] & batch["slot_mask"])[:, :, None]      # [B,S,1]
    step_has_fut = fut_valid.any(dim=-1)                              # [B,S,E]
    sel = focal & step_has_fut                                       # [B,S,E]

    err = torch.linalg.vector_norm(
        traj - fut_pos[:, :, :, None], dim=-1
    )                                                                # [B,S,E,K,N]
    fv = fut_valid[:, :, :, None].to(err.dtype)
    ade_k = (err * fv).sum(-1) / fv.sum(-1).clamp_min(1.0)           # [B,S,E,K]
    kstar = ade_k.argmin(dim=-1)                                     # [B,S,E]

    winner = torch.gather(
        traj, 3, kstar[:, :, :, None, None, None].expand(-1, -1, -1, 1, cfg.horizon, 2)
    ).squeeze(3)                                                     # [B,S,E,N,2]
    # beta 0.1: targets are pos_scale-normalised, so a 5 m error is ~0.1 here —
    # the default beta=1.0 keeps that in the squared regime and starves the
    # trajectory head of gradient once the recon terms saturate (exp1).
    l_traj = _masked(
        F.smooth_l1_loss(winner, fut_pos, reduction="none", beta=0.1),
        sel[:, :, :, None] & fut_valid,
    )
    l_mode = _masked(
        F.cross_entropy(
            out["mode_logits"].flatten(0, 2), kstar.flatten(), reduction="none"
        ).view_as(kstar),
        sel,
    )

    total = (
        cfg.lambda_aux * l_aux + cfg.lambda_next * l_next
        + cfg.lambda_recon_ft * l_recon_ft + cfg.lambda_recon_fupd * l_recon_fupd
        + cfg.lambda_traj * l_traj + cfg.lambda_mode * l_mode
    )
    return {
        "total": total,
        "aux": l_aux.detach(),
        "next": l_next.detach(),
        "recon_ft": l_recon_ft.detach(),
        "recon_fupd": l_recon_fupd.detach(),
        "traj": l_traj.detach(),
        "mode": l_mode.detach(),
    }
