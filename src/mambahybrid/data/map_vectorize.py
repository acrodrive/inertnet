"""Polyline helpers: frame transform + fixed-length resampling."""
from __future__ import annotations

import numpy as np


def rot_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray([[c, -s], [s, c]], np.float32)


def to_anchor_frame(xy: np.ndarray, anchor_xy: np.ndarray, anchor_yaw: float) -> np.ndarray:
    """World xy -> frame with origin at anchor_xy and +x along anchor_yaw."""
    r_inv = rot_matrix(-anchor_yaw)
    return (xy - anchor_xy[None]) @ r_inv.T


def wrap_angle(a: np.ndarray | float):
    return (a + np.pi) % (2 * np.pi) - np.pi


def resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample an [m, 2] polyline to exactly [n, 2] by arc-length interpolation.

    A single-point feature (stop sign) is broadcast.
    """
    pts = np.asarray(pts, np.float32)
    if len(pts) == 0:
        return np.zeros((n, 2), np.float32)
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-6:
        return np.repeat(pts[:1], n, axis=0)
    target = np.linspace(0.0, total, n)
    x = np.interp(target, cum, pts[:, 0])
    y = np.interp(target, cum, pts[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def polyline_min_dist(pts_local: np.ndarray) -> float:
    """Min distance from the anchor origin to any vertex of the polyline."""
    if len(pts_local) == 0:
        return 1e9
    return float(np.min(np.linalg.norm(pts_local, axis=1)))
