"""End-to-end shape + backward test on the real local shard (CPU, tiny config)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mambahybrid.config import Config
from mambahybrid.data import WOMDDataset, collate
from mambahybrid.losses import compute_losses
from mambahybrid.metrics import occlusion_reconstruction_error, trajectory_metrics
from mambahybrid.model import MambaHybrid

SHARD = "dataset/Waymo/training/training.tfrecord-00000-of-01000"


def tiny_cfg() -> Config:
    return Config(
        n_slots=24, max_map_tokens=32, poly_points=12,
        d_model=32, mamba_layers=2, mamba_d_state=8, decoder_layers=1, n_heads=2,
        n_modes=6, horizon=12, warmup_steps=4, min_future_steps=4,
        grad_checkpoint=False, batch_size=2,
    )


def test_end_to_end():
    cfg = tiny_cfg()
    ds = WOMDDataset(SHARD, cfg.preprocess(), split="train", cache_dir=None, seed=0)
    assert len(ds) > 100
    batch = collate([ds[0], ds[1]])

    model = MambaHybrid(cfg)
    model.train()
    out = model(batch)

    B, S, T = batch["gt_valid"].shape
    E = out["elig_idx"].numel()
    assert out["prior_pos_now"].shape == (B, S, T, 3)
    assert out["ft_box"]["pos"].shape == (B, S, T, 3)
    assert out["traj"].shape == (B, S, E, cfg.n_modes, cfg.horizon, 2)
    assert out["mode_logits"].shape == (B, S, E, cfg.n_modes)

    losses = compute_losses(out, batch, cfg)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()

    n_grad = sum(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    n_param = sum(1 for _ in model.parameters())
    assert n_grad == n_param, f"{n_grad}/{n_param} params got finite grads"

    model.eval()
    with torch.no_grad():
        out = model(batch)
    tm = trajectory_metrics(out, batch, cfg)
    re = occlusion_reconstruction_error(out, batch, cfg)
    assert set(tm) == {"all", "occluded", "clean"}
    assert "recon_ade_ft" in re
    print("OK", {k: v["n"] for k, v in tm.items()}, "recon n:", re["n"])


if __name__ == "__main__":
    test_end_to_end()
