"""Embedding, map encoder, correction gate, and the map cross-attention decoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.womd_proto import N_LANE_STATES, N_MAP_LAYERS, N_OBJECT_CLASSES
from ..data.preprocess import OBJ_CONT_DIM


def mlp(d_in: int, d_hidden: int, d_out: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(d_in, d_hidden), nn.GELU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(d_hidden, d_hidden), nn.GELU()]
    layers.append(nn.Linear(d_hidden, d_out))
    return nn.Sequential(*layers)


class ObjectEmbed(nn.Module):
    """Per-step per-slot object token (3.2.2 Embedding(Object))."""

    def __init__(self, d_model: int):
        super().__init__()
        self.cont = mlp(OBJ_CONT_DIM, d_model, d_model)
        self.cls = nn.Embedding(N_OBJECT_CLASSES, d_model)
        self.valid = nn.Embedding(2, d_model)

    def forward(self, cont_t: torch.Tensor, obj_class: torch.Tensor,
                valid_t: torch.Tensor) -> torch.Tensor:
        # cont_t [B,S,10]  obj_class [B,S]  valid_t [B,S] bool
        return (
            self.cont(cont_t)
            + self.cls(obj_class)
            + self.valid(valid_t.long())
        )


class MapEncoder(nn.Module):
    """Static polyline tokens (encoded once) + per-step dynamic signal overlay."""

    def __init__(self, d_model: int, poly_points: int):
        super().__init__()
        self.poly_points = poly_points
        self.point_mlp = mlp(4, d_model, d_model)          # (x, y, dx, dy)
        self.layer_emb = nn.Embedding(N_MAP_LAYERS, d_model)
        self.merge = mlp(d_model, d_model, d_model)
        self.signal_emb = nn.Embedding(N_LANE_STATES, d_model)
        nn.init.zeros_(self.signal_emb.weight)             # UNKNOWN/absent ~ no-op at init

    def encode_static(self, map_points: torch.Tensor, map_layer: torch.Tensor,
                      map_mask: torch.Tensor) -> torch.Tensor:
        # map_points [B,M,P,2]
        d = map_points[:, :, 1:, :] - map_points[:, :, :-1, :]
        d = torch.cat([d, d[:, :, -1:, :]], dim=2)         # [B,M,P,2]
        feat = torch.cat([map_points, d], dim=-1)          # [B,M,P,4]
        h = self.point_mlp(feat)                           # [B,M,P,D]
        h = h.max(dim=2).values                            # [B,M,D]
        h = self.merge(h + self.layer_emb(map_layer))
        return h * map_mask[..., None]

    def dynamic(self, static_tok: torch.Tensor, signal_state_t: torch.Tensor,
                map_mask: torch.Tensor) -> torch.Tensor:
        # signal_state_t [B,M] int
        sig = self.signal_emb(signal_state_t)
        return (static_tok + sig) * map_mask[..., None]


class CorrectionGate(nn.Module):
    """MLP soft-gate: blend observation x_raw and Mamba prior y_t (3.2.3)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.valid_emb = nn.Embedding(2, d_model)
        self.gate = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.Sigmoid())
        self.refine = mlp(d_model, d_model, d_model)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, y_t: torch.Tensor, x_raw_t: torch.Tensor,
                valid_t: torch.Tensor) -> torch.Tensor:
        v = self.valid_emb(valid_t.long())
        g = self.gate(torch.cat([y_t, x_raw_t, v], dim=-1))
        fused = g * x_raw_t + (1.0 - g) * y_t
        return self.norm(fused + self.refine(fused))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = mlp(d_model, ffn_mult * d_model, d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(q)
        a, _ = self.attn(h, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        q = q + a
        q = q + self.ffn(self.norm2(q))
        return q


class MapDecoder(nn.Module):
    """Cross-attention of object slots over {slots, map} (3.2.4)."""

    def __init__(self, d_model: int, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads) for _ in range(n_layers)]
        )

    def forward(self, f_t: torch.Tensor, x_map_t: torch.Tensor,
                slot_mask: torch.Tensor, map_mask: torch.Tensor) -> torch.Tensor:
        kv = torch.cat([f_t, x_map_t], dim=1)                       # [B, S+M, D]
        pad = ~torch.cat([slot_mask, map_mask], dim=1)              # True = ignore
        q = f_t
        for layer in self.layers:
            q = layer(q, kv, pad)
            kv = torch.cat([q, x_map_t], dim=1)
        return q
