"""MambaHybrid: the per-step recurrence (Loop 1) + one-shot future decode.

Per step t  (t = 0 .. T-1), for every slot independently:

    x_raw_t                = ObjectEmbed(obj_cont[t], class, valid[t])
    y_t,  mamba_state      = Mamba.step(f_{t-1}, mamba_state)      # input is prev corrected feat
    f_t                    = CorrectionGate(y_t, x_raw_t, valid[t])
    x_map_t                = MapEncoder.static + signal(t)
    f_t_updated            = MapDecoder(f_t, x_map_t)
    (heads read y_t, f_t, f_t_updated)
    f_{t}  ->  fed back as Mamba input at t+1   (f_t_updated is NOT fed back)

Future trajectories are decoded one-shot from f_t_updated at every eligible step
(warmup <= t <= T-1-min_future), not by rolling the recurrence forward.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .encoders import build_encoder
from .heads import GTHead, PriorHead, TrajectoryHead
from .modules import CorrectionGate, MapDecoder, MapEncoder, ObjectEmbed


class MambaHybrid(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.obj_embed = ObjectEmbed(d)
        self.map_encoder = MapEncoder(d, cfg.poly_points)
        self.encoder = build_encoder(cfg)
        self.gate = CorrectionGate(d)
        self.decoder = MapDecoder(d, n_layers=cfg.decoder_layers, n_heads=cfg.n_heads)
        self.prior_head = PriorHead(d)
        self.gt_head = GTHead(d)
        self.traj_head = TrajectoryHead(d, cfg.n_modes, cfg.horizon)
        # the attn baseline threads a graph-carrying state between steps, which
        # is incompatible with per-step checkpointing.
        self.grad_checkpoint = cfg.grad_checkpoint and cfg.encoder != "attn"

        # The forward is a Python loop over T (~91) steps, each firing hundreds of
        # tiny CUDA kernels -> launch-bound. torch.compile fuses each step's ops
        # (identical shapes every iteration). We compile _step through a plain
        # wrapper rather than compiling it directly: dynamo must be the *outer*
        # layer over torch.utils.checkpoint, not the other way around. Skip for
        # "attn": its state grows each step so it would recompile every t.
        self._compiled_step = None
        if getattr(cfg, "compile", False) and cfg.encoder != "attn":
            self._compiled_step = torch.compile(self._run_step, dynamic=False)

    def _run_step(self, checkpointing: bool, *args):
        if checkpointing:
            return checkpoint(self._step, *args, use_reentrant=False,
                              preserve_rng_state=False)
        return self._step(*args)

    # ------------------------------------------------------------------ step --
    def _step(self, x_raw_t, f_prev, valid_t, static_map, signal_t,
              slot_mask, map_mask, *mamba_state):
        B, S, D = x_raw_t.shape
        y_flat, new_state = self.encoder.step(
            f_prev.reshape(B * S, D), list(mamba_state)
        )
        y_t = y_flat.reshape(B, S, D)
        f_t = self.gate(y_t, x_raw_t, valid_t)
        x_map_t = self.map_encoder.dynamic(static_map, signal_t, map_mask)
        f_upd_t = self.decoder(f_t, x_map_t, slot_mask, map_mask)
        return (y_t, f_t, f_upd_t, *new_state)

    # --------------------------------------------------------------- forward --
    def forward(self, batch: dict) -> dict:
        cfg = self.cfg
        obj_cont = batch["obj_cont"]                 # [B,S,T,10]
        obj_class = batch["obj_class"]               # [B,S]
        obj_valid = batch["obj_valid"]               # [B,S,T]
        slot_mask = batch["slot_mask"]               # [B,S]
        map_mask = batch["map_mask"]                 # [B,M]
        B, S, T, _ = obj_cont.shape
        device, dtype = obj_cont.device, obj_cont.dtype

        static_map = self.map_encoder.encode_static(
            batch["map_points"], batch["map_layer"], map_mask
        )                                            # [B,M,D]

        mamba_state = self.encoder.init_state(B * S, device, dtype)

        prior_now, prior_next = [], []
        ft_boxes = {k: [] for k in ("class_logits", "pos", "heading", "size", "vel")}
        fupd_boxes = {k: [] for k in ("class_logits", "pos", "heading", "size", "vel")}
        traj_list, mode_list, elig_idx = [], [], []

        lo = cfg.warmup_steps
        hi = T - 1 - cfg.min_future_steps

        f_prev = None
        for t in range(T):
            x_raw_t = self.obj_embed(obj_cont[:, :, t], obj_class, obj_valid[:, :, t])
            if f_prev is None:
                f_prev = x_raw_t                     # first step: seed with observation
            valid_t = obj_valid[:, :, t]

            args = (x_raw_t, f_prev, valid_t, static_map,
                    batch["signal_state"][:, t], slot_mask, map_mask, *mamba_state)
            ckpt = self.grad_checkpoint and self.training
            if self._compiled_step is not None:
                ret = self._compiled_step(ckpt, *args)
            elif ckpt:
                ret = checkpoint(self._step, *args, use_reentrant=False,
                                 preserve_rng_state=False)
            else:
                ret = self._step(*args)
            y_t, f_t, f_upd_t = ret[0], ret[1], ret[2]
            mamba_state = list(ret[3:])

            pr = self.prior_head(y_t)
            prior_now.append(pr["pos_now"])
            prior_next.append(pr["pos_next"])
            bft = self.gt_head(f_t)
            bfu = self.gt_head(f_upd_t)
            for k in ft_boxes:
                ft_boxes[k].append(bft[k])
                fupd_boxes[k].append(bfu[k])

            if lo <= t <= hi:
                th = self.traj_head(f_upd_t)
                traj_list.append(th["traj"])
                mode_list.append(th["mode_logits"])
                elig_idx.append(t)

            f_prev = f_t

        def stack_t(lst):
            return torch.stack(lst, dim=2)           # [B,S,T,...]

        out = {
            "prior_pos_now": stack_t(prior_now),                 # [B,S,T,3]
            "prior_pos_next": stack_t(prior_next),               # [B,S,T,3]
            "ft_box": {k: stack_t(v) for k, v in ft_boxes.items()},
            "fupd_box": {k: stack_t(v) for k, v in fupd_boxes.items()},
            "traj": torch.stack(traj_list, dim=2),               # [B,S,E,K,N,2]
            "mode_logits": torch.stack(mode_list, dim=2),        # [B,S,E,K]
            "elig_idx": torch.tensor(elig_idx, device=device),   # [E]
        }
        return out
