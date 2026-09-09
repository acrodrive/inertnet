"""Output heads (3.2.5)."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.womd_proto import N_OBJECT_CLASSES
from .modules import mlp


class PriorHead(nn.Module):
    """Reads the Mamba prior y_t. Supervises the physical transition:
    L_aux -> current position, L_next -> next-step position (3.2.1)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.now = mlp(d_model, d_model, 3)
        self.nxt = mlp(d_model, d_model, 3)

    def forward(self, y_t: torch.Tensor) -> dict:
        return {"pos_now": self.now(y_t), "pos_next": self.nxt(y_t)}


class GTHead(nn.Module):
    """Full box state from a corrected feature (used on both f_t and f_t_updated)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.trunk = mlp(d_model, d_model, d_model)
        self.cls = nn.Linear(d_model, N_OBJECT_CLASSES)
        self.pos = nn.Linear(d_model, 3)
        self.heading = nn.Linear(d_model, 2)     # (sin, cos)
        self.size = nn.Linear(d_model, 3)
        self.vel = nn.Linear(d_model, 2)

    def forward(self, f: torch.Tensor) -> dict:
        h = self.trunk(f)
        return {
            "class_logits": self.cls(h),
            "pos": self.pos(h),
            "heading": self.heading(h),
            "size": self.size(h),
            "vel": self.vel(h),
        }


class TrajectoryHead(nn.Module):
    """One-shot multi-modal future decode from f_t_updated (K modes x N steps).

    The offsets are regressed in the agent's own local frame (origin + x-axis =
    the agent's pose at the decode step), then rotated back into the shared scene
    frame so the loss and metrics — which live in scene space — are unchanged.
    In the local frame "go straight" is the same target for every agent whatever
    its heading, so a single head can share motion structure across all slots.
    """

    def __init__(self, d_model: int, n_modes: int, horizon: int):
        super().__init__()
        self.k = n_modes
        self.n = horizon
        self.trunk = mlp(d_model, 2 * d_model, d_model)
        self.traj = nn.Linear(d_model, n_modes * horizon * 2)
        self.mode = nn.Linear(d_model, n_modes)

    def forward(self, f_upd: torch.Tensor, anchor_pos: torch.Tensor,
                anchor_yaw: torch.Tensor) -> dict:
        # anchor_pos [B,S,2], anchor_yaw [B,S] — the slot's pose (scene frame,
        # pos_scale-normalised) at the step this trajectory is decoded from.
        h = self.trunk(f_upd)
        b, s, _ = h.shape
        local = self.traj(h).view(b, s, self.k, self.n, 2)      # agent frame
        cos, sin = torch.cos(anchor_yaw), torch.sin(anchor_yaw)  # [B,S]
        cos, sin = cos[:, :, None, None], sin[:, :, None, None]
        lx, ly = local[..., 0], local[..., 1]
        traj = torch.stack([cos * lx - sin * ly, sin * lx + cos * ly], dim=-1)
        traj = traj + anchor_pos[:, :, None, None, :]           # -> scene frame
        return {"traj": traj, "mode_logits": self.mode(h)}
