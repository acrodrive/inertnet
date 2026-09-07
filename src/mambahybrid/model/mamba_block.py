"""Selective SSM (Mamba / S6) in pure PyTorch with a recurrent ``step`` API.

The hybrid model feeds each step's corrected feature ``f_{t-1}`` back in as the
next input, so the sequence cannot be scanned in parallel -- we must step. This
implementation is therefore written step-first. On a CUDA box you may swap in
``mamba_ssm.Mamba(...).step(...)``; the interface here matches (returns ``y_t``
and the updated ``(conv_state, ssm_state)``).

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
Spaces" (2023), Algorithm 2.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # softplus(dt_proj.bias) ~ dt in [dt_min, dt_max]
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)
            ).clamp_min(1e-4)
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def init_state(self, n: int, device, dtype):
        conv_state = torch.zeros(n, self.d_inner, self.d_conv, device=device, dtype=dtype)
        ssm_state = torch.zeros(n, self.d_inner, self.d_state, device=device, dtype=dtype)
        return conv_state, ssm_state

    def step(self, x_t: torch.Tensor, state):
        """x_t: [N, d_model].  state: (conv_state [N,d_inner,d_conv], ssm_state [N,d_inner,d_state]).

        Runs with autocast disabled: the selective-scan recurrence is unstable
        in fp16/bf16.
        """
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            conv_state, ssm_state = state
            xz = self.in_proj(x_t.float())                          # [N, 2*d_inner]
            x, z = xz.chunk(2, dim=-1)

            conv_state = torch.roll(conv_state, shifts=-1, dims=-1).clone()
            conv_state[:, :, -1] = x
            x_conv = torch.einsum("ndk,dk->nd", conv_state, self.conv1d.weight.squeeze(1))
            x = F.silu(x_conv + self.conv1d.bias)                   # [N, d_inner]

            dbl = self.x_proj(x)                                    # [N, dt_rank + 2*d_state]
            dt, B, C = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = F.softplus(self.dt_proj(dt))                       # [N, d_inner]

            A = -torch.exp(self.A_log)                              # [d_inner, d_state]
            dA = torch.exp(dt[:, :, None] * A[None])                # [N, d_inner, d_state]
            dB_x = dt[:, :, None] * B[:, None, :] * x[:, :, None]
            ssm_state = dA * ssm_state + dB_x
            y = torch.einsum("nds,ns->nd", ssm_state, C) + self.D[None] * x
            y = y * F.silu(z)
            y = self.out_proj(y)
        return y, (conv_state, ssm_state)


class MambaStack(nn.Module):
    """n_layers of (RMSNorm -> MambaBlock -> residual), stepped together.

    State is a *flat* list [conv0, ssm0, conv1, ssm1, ...] so it threads cleanly
    through ``torch.utils.checkpoint``.
    """

    def __init__(self, d_model: int, n_layers: int = 2, **block_kw):
        super().__init__()
        self.layers = nn.ModuleList(
            [MambaBlock(d_model, **block_kw) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(n_layers)])

    def init_state(self, n: int, device, dtype) -> list[torch.Tensor]:
        flat: list[torch.Tensor] = []
        for blk in self.layers:
            conv, ssm = blk.init_state(n, device, dtype)
            flat += [conv, ssm]
        return flat

    def step(self, x_t: torch.Tensor, flat_state: list[torch.Tensor]):
        new_state: list[torch.Tensor] = []
        h = x_t
        for i, (blk, norm) in enumerate(zip(self.layers, self.norms)):
            st = (flat_state[2 * i], flat_state[2 * i + 1])
            y, (conv, ssm) = blk.step(norm(h), st)
            h = h + y
            new_state += [conv, ssm]
        return h, new_state
