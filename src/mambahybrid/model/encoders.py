"""Temporal encoder variants for the ablation (Specification 3.4.2).

All expose the same stepwise interface so the hybrid time-loop is unchanged:

    state = enc.init_state(n, device, dtype)
    y_t, state = enc.step(x_t, state)        # x_t: [N, d_model]

- "mamba" : selective SSM (the proposed model)   -> MambaStack
- "gru"   : stacked GRU  (recurrent baseline C)
- "attn"  : causal self-attention over past inputs (Transformer-only baseline B)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .mamba_block import MambaStack


class GRUEncoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 4):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.n_layers = n_layers
        self.d_model = d_model

    def init_state(self, n, device, dtype):
        return [torch.zeros(self.n_layers, n, self.d_model, device=device, dtype=dtype)]

    def step(self, x_t, state):
        h = state[0]
        y, h = self.gru(x_t[:, None], h)
        return y[:, 0], [h]


class CausalAttnEncoder(nn.Module):
    """Transformer-only: at step t, attend over all past step inputs (causal)."""

    def __init__(self, d_model: int, n_layers: int = 4, n_heads: int = 4, max_len: int = 128):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, batch_first=True,
                                       norm_first=True, activation="gelu")
            for _ in range(n_layers)
        ])
        self.pos = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        self.d_model = d_model

    def init_state(self, n, device, dtype):
        return [torch.zeros(n, 0, self.d_model, device=device, dtype=dtype)]

    def step(self, x_t, state):
        past = state[0]
        seq = torch.cat([past, x_t[:, None]], dim=1)          # [N, t+1, D]
        t = seq.size(1)
        h = seq + self.pos[:t][None]
        for layer in self.layers:
            # hybrid.py's outer per-step checkpoint is disabled for "attn"
            # (the growing seq length breaks its replay), but that leaves the
            # whole re-run-every-step transformer stack un-checkpointed ->
            # O(T^2) *activation memory*, not just O(T^2) compute. Checkpoint
            # each layer call instead: same recompute cost, but only this
            # call's activations are live at once. Safe across the growing
            # sequence since each checkpoint call is self-contained (no state
            # spans steps here, unlike the outer loop).
            if self.training and h.requires_grad:
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        # keep the raw inputs (not activations) as state -> full BPTT, O(T^2)
        # compute. This cost is exactly what the thesis argues against.
        return h[:, -1], [seq]


def build_encoder(cfg):
    if cfg.encoder == "mamba":
        return MambaStack(
            cfg.d_model, n_layers=cfg.mamba_layers, d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv, expand=cfg.mamba_expand,
        )
    if cfg.encoder == "gru":
        return GRUEncoder(cfg.d_model, n_layers=cfg.mamba_layers)
    if cfg.encoder == "attn":
        return CausalAttnEncoder(cfg.d_model, n_layers=cfg.mamba_layers, n_heads=cfg.n_heads)
    raise ValueError(f"unknown encoder {cfg.encoder!r}")
