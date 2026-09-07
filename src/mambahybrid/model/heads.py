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
    """One-shot multi-modal future decode from f_t_updated (K modes x N steps)."""

    def __init__(self, d_model: int, n_modes: int, horizon: int):
        super().__init__()
        self.k = n_modes
        self.n = horizon
        self.trunk = mlp(d_model, 2 * d_model, d_model)
        self.traj = nn.Linear(d_model, n_modes * horizon * 2)
        self.mode = nn.Linear(d_model, n_modes)

    def forward(self, f_upd: torch.Tensor) -> dict:
        h = self.trunk(f_upd)
        b, s, _ = h.shape
        traj = self.traj(h).view(b, s, self.k, self.n, 2)
        return {"traj": traj, "mode_logits": self.mode(h)}
