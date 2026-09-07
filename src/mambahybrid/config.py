"""Single config object for data, model, loss and training. YAML-loadable."""
from __future__ import annotations

import dataclasses

import yaml

from .data.preprocess import PreprocessConfig


@dataclasses.dataclass
class Config:
    # ---- data / preprocess ----
    train_shards: str = "dataset/Waymo/training/training.tfrecord-000*-of-01000"
    val_shards: str = "dataset/Waymo/training/training.tfrecord-0009*-of-01000"
    cache_dir: str | None = "cache"
    n_slots: int = 300
    fov_radius: float = 100.0
    max_map_tokens: int = 256
    poly_points: int = 20
    pos_scale: float = 50.0
    vel_scale: float = 10.0
    size_scale: float = 10.0
    occlusion_prob: float = 0.5
    occlusion_max_gaps: int = 2
    random_rotate: bool = True
    subtract_ego_velocity: bool = False

    # ---- model ----
    encoder: str = "mamba"     # "mamba" | "gru" | "attn"  (ablation, spec 3.4.2)
    d_model: int = 128
    mamba_layers: int = 4
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    decoder_layers: int = 2
    n_heads: int = 4
    n_modes: int = 6
    horizon: int = 50          # future steps predicted from each eligible t (5.0 s)
    warmup_steps: int = 10     # no losses / no prediction before this step
    min_future_steps: int = 10 # need at least this many future steps to predict
    grad_checkpoint: bool = True

    # ---- loss weights (3.2.6) ----
    lambda_aux: float = 0.5
    lambda_next: float = 0.5
    lambda_recon_ft: float = 1.0
    lambda_recon_fupd: float = 1.0
    lambda_traj: float = 1.0
    lambda_mode: float = 0.5

    # ---- optim / train ----
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int = 100_000
    warmup_iters: int = 2_000
    log_every: int = 50
    val_every: int = 2_000
    ckpt_dir: str = "runs/exp1"
    num_workers: int = 6
    seed: int = 0
    amp: bool = True
    device: str = "cuda"

    def preprocess(self) -> PreprocessConfig:
        return PreprocessConfig(
            n_slots=self.n_slots,
            fov_radius=self.fov_radius,
            max_map_tokens=self.max_map_tokens,
            poly_points=self.poly_points,
            pos_scale=self.pos_scale,
            vel_scale=self.vel_scale,
            size_scale=self.size_scale,
            occlusion_prob=self.occlusion_prob,
            occlusion_max_gaps=self.occlusion_max_gaps,
            random_rotate=self.random_rotate,
            subtract_ego_velocity=self.subtract_ego_velocity,
        )

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def dump(self, path: str) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(dataclasses.asdict(self), fh, sort_keys=False)
