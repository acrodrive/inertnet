"""Scenario -> training sample.

One WOMD scenario becomes one sample. All geometry is expressed in a single
anchor frame (SDC pose at ``current_time_index``); the frame is never re-centred
per step, so a constant-velocity object traces a straight line in sample space.

The sample keeps the *full* 91-step sequence. Dense per-step supervision (state
reconstruction at every step + trajectory prediction from every step) replaces
the leaderboard's single history/future cut.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import womd_proto as P
from .map_vectorize import (
    polyline_min_dist,
    resample_polyline,
    to_anchor_frame,
    wrap_angle,
)
from .parser import Scenario

# continuous per-step object features: x y z sin(h) cos(h) L W H vx vy
OBJ_CONT_DIM = 10


@dataclasses.dataclass
class PreprocessConfig:
    n_slots: int = 300
    fov_radius: float = 100.0
    max_map_tokens: int = 256
    poly_points: int = 20
    # normalisation (features & targets live in this scaled space; metrics unscale)
    pos_scale: float = 50.0
    vel_scale: float = 10.0
    size_scale: float = 10.0
    # synthetic occlusion
    occlusion_prob: float = 0.5          # per eligible track
    occlusion_max_gaps: int = 2
    gap_len_lognorm_mean: float = 1.1    # ln(steps); exp(1.1)~3 -> median ~0.3s
    gap_len_lognorm_sigma: float = 0.6
    gap_len_max: int = 22
    occlude_focal_only: bool = False
    # augmentation
    random_rotate: bool = True
    subtract_ego_velocity: bool = False


def _agent_slot_order(sc: Scenario, cfg: PreprocessConfig, anchor_xy, anchor_yaw) -> np.ndarray:
    """Return the ordered agent indices that fill the slots."""
    A = sc.num_agents
    focal = [int(i) for i in sc.tracks_to_predict if 0 <= int(i) < A]
    focal_set = set(focal)

    # min distance to anchor origin over valid steps (fallback: +inf)
    xy_local = to_anchor_frame(sc.xyz[:, :, :2].reshape(-1, 2), anchor_xy, anchor_yaw)
    xy_local = xy_local.reshape(A, sc.num_steps, 2)
    dist = np.linalg.norm(xy_local, axis=2)
    dist = np.where(sc.valid, dist, np.inf)
    min_dist = dist.min(axis=1)                       # [A]
    first_seen = np.argmax(sc.valid, axis=1)          # [A]
    first_seen = np.where(sc.valid.any(axis=1), first_seen, sc.num_steps)

    others = [
        i for i in range(A)
        if i not in focal_set and min_dist[i] <= cfg.fov_radius
    ]
    others.sort(key=lambda i: (first_seen[i], min_dist[i]))

    order = focal + others
    return np.asarray(order[: cfg.n_slots], np.int64)


def _sample_gaps(rng, valid_steps: np.ndarray, cfg: PreprocessConfig) -> list[tuple[int, int]]:
    """valid_steps: sorted array of step indices where the agent is observed."""
    if len(valid_steps) < 6:
        return []
    lo, hi = int(valid_steps[0]), int(valid_steps[-1])
    span = hi - lo
    if span < 5:
        return []
    n_gaps = rng.integers(1, cfg.occlusion_max_gaps + 1)
    gaps = []
    for _ in range(int(n_gaps)):
        glen = int(np.clip(round(rng.lognormal(cfg.gap_len_lognorm_mean,
                                               cfg.gap_len_lognorm_sigma)),
                           1, min(cfg.gap_len_max, span - 2)))
        start = int(rng.integers(lo + 1, hi - glen)) if hi - glen > lo + 1 else lo + 1
        gaps.append((start, start + glen))
    return gaps


def build_sample(sc: Scenario, cfg: PreprocessConfig, rng: np.random.Generator,
                 split: str = "train") -> dict:
    T = sc.num_steps
    S = cfg.n_slots
    M = cfg.max_map_tokens
    Pn = cfg.poly_points
    ta = int(sc.current_time_index)
    sdc = int(sc.sdc_track_index)

    anchor_xy = sc.xyz[sdc, ta, :2].astype(np.float32).copy()
    anchor_yaw = float(sc.heading[sdc, ta])
    if cfg.random_rotate and split == "train":
        anchor_yaw += float(rng.uniform(-np.pi, np.pi))
    ego_vel = sc.velocity[sdc, ta].astype(np.float32) if cfg.subtract_ego_velocity else np.zeros(2, np.float32)

    order = _agent_slot_order(sc, cfg, anchor_xy, anchor_yaw)
    n_real = len(order)

    slot_mask = np.zeros((S,), bool)
    slot_mask[:n_real] = True
    is_focal = np.zeros((S,), bool)
    focal_set = {int(i) for i in sc.tracks_to_predict}
    for s, a in enumerate(order):
        is_focal[s] = int(a) in focal_set

    obj_class = np.zeros((S,), np.int64)
    pos = np.zeros((S, T, 3), np.float32)
    hdg = np.zeros((S, T), np.float32)
    size = np.zeros((S, T, 3), np.float32)
    vel = np.zeros((S, T, 2), np.float32)
    valid = np.zeros((S, T), bool)

    cos_a, sin_a = np.cos(-anchor_yaw), np.sin(-anchor_yaw)
    R_inv = np.asarray([[cos_a, -sin_a], [sin_a, cos_a]], np.float32)

    for s, a in enumerate(order):
        v = sc.valid[a]
        valid[s] = v
        obj_class[s] = sc.object_types[a]
        p_local = (sc.xyz[a, :, :2] - anchor_xy[None]) @ R_inv.T
        pos[s, :, :2] = np.where(v[:, None], p_local, 0.0)
        pos[s, :, 2] = np.where(v, sc.xyz[a, :, 2] - sc.xyz[sdc, ta, 2], 0.0)
        hdg[s] = np.where(v, wrap_angle(sc.heading[a] - anchor_yaw), 0.0)
        size[s] = np.where(v[:, None], sc.size_lwh[a], 0.0)
        vv = (sc.velocity[a] - ego_vel[None]) @ R_inv.T
        vel[s] = np.where(v[:, None], vv, 0.0)

    # normalise geometry (targets stay in this space; metrics multiply back)
    pos[..., :2] /= cfg.pos_scale
    pos[..., 2] /= cfg.pos_scale
    vel /= cfg.vel_scale
    size /= cfg.size_scale

    recon_mask = valid.copy() & slot_mask[:, None]        # original observability
    synth_mask = np.zeros((S, T), bool)

    # ---- synthetic occlusion (train + val, never on the GT copies) ----
    in_valid = valid.copy()
    in_pos, in_hdg, in_size, in_vel = pos.copy(), hdg.copy(), size.copy(), vel.copy()
    if split in ("train", "val"):
        for s in range(n_real):
            if cfg.occlude_focal_only and not is_focal[s]:
                continue
            if rng.random() > cfg.occlusion_prob:
                continue
            vs = np.where(valid[s])[0]
            for g0, g1 in _sample_gaps(rng, vs, cfg):
                in_valid[s, g0:g1] = False
                in_pos[s, g0:g1] = 0.0
                in_hdg[s, g0:g1] = 0.0
                in_size[s, g0:g1] = 0.0
                in_vel[s, g0:g1] = 0.0
                synth_mask[s, g0:g1] = True

    # continuous input features (post-occlusion)
    obj_cont = np.concatenate([
        in_pos,
        np.sin(in_hdg)[..., None], np.cos(in_hdg)[..., None],
        in_size, in_vel,
    ], axis=-1).astype(np.float32)                        # [S, T, 10]

    # ---- map ----
    polys = sc.polylines
    scored = []
    for pl in polys:
        loc = to_anchor_frame(pl.points[:, :2], anchor_xy, anchor_yaw)
        scored.append((polyline_min_dist(loc), pl, loc))
    scored.sort(key=lambda x: x[0])
    scored = [x for x in scored if x[0] <= cfg.fov_radius * 1.5][:M]

    map_points = np.zeros((M, Pn, 2), np.float32)
    map_layer = np.zeros((M,), np.int64)
    map_mask = np.zeros((M,), bool)
    map_lane_id = np.full((M,), -1, np.int64)
    for i, (_, pl, loc) in enumerate(scored):
        map_points[i] = resample_polyline(loc, Pn) / cfg.pos_scale
        map_layer[i] = pl.layer
        map_mask[i] = True
        if pl.layer <= P.LAYER_LANE_BIKE:
            map_lane_id[i] = pl.feature_id

    # per-step signal state per map token (UNKNOWN=0 elsewhere)
    signal_state = np.zeros((T, M), np.int64)
    lane_to_slot = {int(lid): i for i, lid in enumerate(map_lane_id) if lid >= 0}
    for t in range(T):
        for lid, st in zip(sc.signal_lane_ids[t], sc.signal_states[t]):
            j = lane_to_slot.get(int(lid))
            if j is not None:
                signal_state[t, j] = int(st)

    return dict(
        scenario_id=sc.scenario_id,
        current_time_index=np.int64(ta),
        # model inputs
        obj_cont=obj_cont,                # [S,T,10]
        obj_class=obj_class,              # [S]
        obj_valid=in_valid,               # [S,T] bool (post-occlusion)
        slot_mask=slot_mask,              # [S]
        is_focal=is_focal,                # [S]
        map_points=map_points,            # [M,Pn,2]
        map_layer=map_layer,              # [M]
        map_mask=map_mask,                # [M]
        signal_state=signal_state,        # [T,M]
        # supervision targets (pre-occlusion)
        gt_pos=pos,                       # [S,T,3]
        gt_heading=hdg,                   # [S,T]
        gt_size=size,                     # [S,T,3]
        gt_vel=vel,                       # [S,T,2]
        gt_valid=valid,                   # [S,T] bool (true observability)
        recon_mask=recon_mask,            # [S,T] bool
        synth_mask=synth_mask,            # [S,T] bool
    )
