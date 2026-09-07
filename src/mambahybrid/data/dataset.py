"""Torch dataset over WOMD shards with byte-offset indexing + sample cache."""
from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from . import womd_proto as P
from .parser import parse_scenario
from .preprocess import PreprocessConfig, build_sample

_ARRAY_KEYS = (
    "obj_cont", "obj_class", "obj_valid", "slot_mask", "is_focal",
    "map_points", "map_layer", "map_mask", "signal_state",
    "gt_pos", "gt_heading", "gt_size", "gt_vel", "gt_valid",
    "recon_mask", "synth_mask", "current_time_index",
)


class WOMDDataset(Dataset):
    """One sample per scenario. Records are addressed by byte offset (built once
    per shard, cached next to the shard), so ``__getitem__`` seeks directly.
    Built samples are cached as ``.npz`` keyed by shard + record + config hash.
    """

    def __init__(self, shard_glob: str, cfg: PreprocessConfig, split: str = "train",
                 cache_dir: str | None = None, seed: int = 0):
        self.shards = sorted(glob.glob(shard_glob))
        if not self.shards:
            raise FileNotFoundError(f"no shards match {shard_glob!r}")
        self.cfg = cfg
        self.split = split
        self.seed = seed
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self._cfg_hash = hashlib.sha1(repr(cfg).encode()).hexdigest()[:8]

        self._offsets: list[list[int]] = [self._shard_offsets(s) for s in self.shards]
        self._items = [
            (si, ri) for si, offs in enumerate(self._offsets) for ri in range(len(offs))
        ]

    @staticmethod
    def _shard_offsets(shard: str) -> list[int]:
        idx_path = shard + ".offidx.json"
        if os.path.exists(idx_path) and os.path.getmtime(idx_path) >= os.path.getmtime(shard):
            with open(idx_path) as fh:
                return json.load(fh)
        offs = P.record_offsets(shard)
        try:
            with open(idx_path, "w") as fh:
                json.dump(offs, fh)
        except OSError:
            pass
        return offs

    def __len__(self) -> int:
        return len(self._items)

    def _sample_cache(self, si: int, ri: int) -> str | None:
        if not self.cache_dir:
            return None
        name = os.path.basename(self.shards[si])
        return os.path.join(
            self.cache_dir, f"{name}.r{ri:05d}.{self._cfg_hash}.{self.split}.npz"
        )

    def __getitem__(self, idx: int) -> dict:
        si, ri = self._items[idx]
        cpath = self._sample_cache(si, ri)
        if cpath and os.path.exists(cpath):
            data = np.load(cpath, allow_pickle=True)
            return {k: self._to_tensor(data[k]) for k in _ARRAY_KEYS}

        payload = P.read_record_at(self.shards[si], self._offsets[si][ri])
        sc = parse_scenario(payload)
        h = hashlib.sha1(f"{sc.scenario_id}/{self.split}/{self.seed}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:7], "little"))
        sample = build_sample(sc, self.cfg, rng, self.split)
        if cpath:
            try:
                np.savez_compressed(cpath, **{k: np.asarray(sample[k]) for k in _ARRAY_KEYS})
            except OSError:
                pass
        return {k: self._to_tensor(sample[k]) for k in _ARRAY_KEYS}

    @staticmethod
    def _to_tensor(arr):
        arr = np.asarray(arr)
        if arr.dtype == bool:
            return torch.from_numpy(arr.astype(np.bool_))
        if np.issubdtype(arr.dtype, np.integer):
            return torch.from_numpy(arr.astype(np.int64))
        return torch.from_numpy(arr.astype(np.float32))


def collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in _ARRAY_KEYS}
