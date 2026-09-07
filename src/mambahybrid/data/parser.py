"""Parse a WOMD Scenario protobuf into plain numpy arrays."""
from __future__ import annotations

import dataclasses

import blackboxprotobuf
import numpy as np

from . import womd_proto as P


@dataclasses.dataclass
class Polyline:
    layer: int                # LAYER_* semantic id
    points: np.ndarray        # [n, 3] float32, global frame
    feature_id: int
    lane_type: int = 0        # for lanes only


@dataclasses.dataclass
class Scenario:
    scenario_id: str
    timestamps: np.ndarray            # [T] float64 seconds
    current_time_index: int
    sdc_track_index: int
    tracks_to_predict: np.ndarray     # [P] int, indices into `tracks`

    track_ids: np.ndarray            # [A] int
    object_types: np.ndarray         # [A] int  (P.OBJECT_TYPES)
    # per-object per-timestep state, global frame
    xyz: np.ndarray                  # [A, T, 3] float32
    heading: np.ndarray              # [A, T] float32
    size_lwh: np.ndarray             # [A, T, 3] float32  (length, width, height)
    velocity: np.ndarray             # [A, T, 2] float32
    valid: np.ndarray                # [A, T] bool

    polylines: list[Polyline]
    # per-timestep traffic-signal state, keyed by controlled lane feature id
    signal_lane_ids: list[np.ndarray]    # len T, each [k] int
    signal_states: list[np.ndarray]      # len T, each [k] int  (P.LANE_STATE_*)
    signal_points: list[np.ndarray]      # len T, each [k, 3] float32

    @property
    def num_steps(self) -> int:
        return len(self.timestamps)

    @property
    def num_agents(self) -> int:
        return len(self.track_ids)


def _decode_states(states_raw: list) -> dict:
    T = len(states_raw)
    xyz = np.zeros((T, 3), np.float32)
    heading = np.zeros((T,), np.float32)
    size = np.zeros((T, 3), np.float32)
    vel = np.zeros((T, 2), np.float32)
    valid = np.zeros((T,), bool)
    for t, s in enumerate(states_raw):
        if not isinstance(s, dict):
            continue
        valid[t] = bool(s.get(P.OS_VALID, 0))
        if not valid[t]:
            continue
        xyz[t, 0] = P.as_f64(s.get(P.OS_CENTER_X, 0.0))
        xyz[t, 1] = P.as_f64(s.get(P.OS_CENTER_Y, 0.0))
        xyz[t, 2] = P.as_f64(s.get(P.OS_CENTER_Z, 0.0))
        heading[t] = P.as_f32(s.get(P.OS_HEADING, 0.0))
        size[t, 0] = P.as_f32(s.get(P.OS_LENGTH, 0.0))
        size[t, 1] = P.as_f32(s.get(P.OS_WIDTH, 0.0))
        size[t, 2] = P.as_f32(s.get(P.OS_HEIGHT, 0.0))
        vel[t, 0] = P.as_f32(s.get(P.OS_VELOCITY_X, 0.0))
        vel[t, 1] = P.as_f32(s.get(P.OS_VELOCITY_Y, 0.0))
    return dict(xyz=xyz, heading=heading, size=size, vel=vel, valid=valid)


def _lane_layer(lane_type: int) -> int:
    return {
        1: P.LAYER_LANE_FREEWAY,
        2: P.LAYER_LANE_SURFACE,
        3: P.LAYER_LANE_BIKE,
    }.get(int(lane_type), P.LAYER_LANE_SURFACE)


def _decode_polyline(points_raw: list) -> np.ndarray:
    pts = [P.point_xyz(mp) for mp in P.as_list(points_raw) if isinstance(mp, dict)]
    if not pts:
        return np.zeros((0, 3), np.float32)
    return np.asarray(pts, np.float32)


def _decode_map(map_features_raw: list) -> list[Polyline]:
    out: list[Polyline] = []
    for mf in P.as_list(map_features_raw):
        if not isinstance(mf, dict):
            continue
        fid = int(mf.get(P.MF_ID, -1))
        if P.MF_LANE in mf:
            lc = mf[P.MF_LANE]
            lt = int(lc.get(P.LC_TYPE, 2))
            pts = _decode_polyline(lc.get(P.LC_POLYLINE))
            if len(pts):
                out.append(Polyline(_lane_layer(lt), pts, fid, lt))
        elif P.MF_ROAD_LINE in mf:
            pts = _decode_polyline(mf[P.MF_ROAD_LINE].get(P.RL_POLYLINE))
            if len(pts):
                out.append(Polyline(P.LAYER_ROAD_LINE, pts, fid))
        elif P.MF_ROAD_EDGE in mf:
            pts = _decode_polyline(mf[P.MF_ROAD_EDGE].get(P.RL_POLYLINE))
            if len(pts):
                out.append(Polyline(P.LAYER_ROAD_EDGE, pts, fid))
        elif P.MF_CROSSWALK in mf:
            pts = _decode_polyline(mf[P.MF_CROSSWALK].get(P.POLY_POLYGON))
            if len(pts):
                out.append(Polyline(P.LAYER_CROSSWALK, pts, fid))
        elif P.MF_SPEED_BUMP in mf:
            pts = _decode_polyline(mf[P.MF_SPEED_BUMP].get(P.POLY_POLYGON))
            if len(pts):
                out.append(Polyline(P.LAYER_SPEED_BUMP, pts, fid))
        elif P.MF_DRIVEWAY in mf:
            pts = _decode_polyline(mf[P.MF_DRIVEWAY].get(P.POLY_POLYGON))
            if len(pts):
                out.append(Polyline(P.LAYER_DRIVEWAY, pts, fid))
        elif P.MF_STOP_SIGN in mf:
            pos = mf[P.MF_STOP_SIGN].get(P.SS_POSITION)
            if isinstance(pos, dict):
                pt = np.asarray([P.point_xyz(pos)], np.float32)
                out.append(Polyline(P.LAYER_STOP_SIGN, pt, fid))
    return out


def _decode_signals(dms_raw: list, T: int):
    lane_ids, states, points = [], [], []
    dms_list = P.as_list(dms_raw)
    for t in range(T):
        dms = dms_list[t] if t < len(dms_list) else None
        ids_t, st_t, pt_t = [], [], []
        if isinstance(dms, dict):
            for ls in P.as_list(dms.get(P.DMS_LANE_STATES)):
                if not isinstance(ls, dict):
                    continue
                ids_t.append(int(ls.get(P.TS_LANE, -1)))
                st_t.append(int(ls.get(P.TS_STATE, 0)))
                sp = ls.get(P.TS_STOP_POINT)
                pt_t.append(P.point_xyz(sp) if isinstance(sp, dict) else (0.0, 0.0, 0.0))
        lane_ids.append(np.asarray(ids_t, np.int64))
        states.append(np.asarray(st_t, np.int64))
        points.append(np.asarray(pt_t, np.float32).reshape(-1, 3))
    return lane_ids, states, points


def parse_scenario(payload: bytes) -> Scenario:
    msg, _ = blackboxprotobuf.decode_message(payload)

    ts = np.asarray([P.as_f64(x) for x in P.as_list(msg.get(P.SC_TIMESTAMPS))], np.float64)
    T = len(ts)
    cti = int(msg.get(P.SC_CURRENT_TIME_INDEX, T // 2))
    sdc = int(msg.get(P.SC_SDC_TRACK_INDEX, 0))

    ttp = []
    for rp in P.as_list(msg.get(P.SC_TRACKS_TO_PREDICT)):
        if isinstance(rp, dict) and P.RP_TRACK_INDEX in rp:
            ttp.append(int(rp[P.RP_TRACK_INDEX]))

    tracks_raw = P.as_list(msg.get(P.SC_TRACKS))
    A = len(tracks_raw)
    track_ids = np.zeros((A,), np.int64)
    obj_types = np.zeros((A,), np.int64)
    xyz = np.zeros((A, T, 3), np.float32)
    heading = np.zeros((A, T), np.float32)
    size = np.zeros((A, T, 3), np.float32)
    vel = np.zeros((A, T, 2), np.float32)
    valid = np.zeros((A, T), bool)
    for a, tr in enumerate(tracks_raw):
        if not isinstance(tr, dict):
            continue
        track_ids[a] = int(tr.get(P.TR_ID, -1))
        obj_types[a] = int(tr.get(P.TR_OBJECT_TYPE, 0))
        st = _decode_states(P.as_list(tr.get(P.TR_STATES)))
        xyz[a] = st["xyz"]
        heading[a] = st["heading"]
        size[a] = st["size"]
        vel[a] = st["vel"]
        valid[a] = st["valid"]

    polylines = _decode_map(msg.get(P.SC_MAP_FEATURES))
    sig_ids, sig_states, sig_points = _decode_signals(msg.get(P.SC_DYNAMIC_MAP_STATES), T)

    sid = msg.get(P.SC_SCENARIO_ID, b"")
    if isinstance(sid, bytes):
        sid = sid.decode("utf-8", "replace")

    return Scenario(
        scenario_id=str(sid),
        timestamps=ts,
        current_time_index=cti,
        sdc_track_index=sdc,
        tracks_to_predict=np.asarray(ttp, np.int64),
        track_ids=track_ids,
        object_types=obj_types,
        xyz=xyz,
        heading=heading,
        size_lwh=size,
        velocity=vel,
        valid=valid,
        polylines=polylines,
        signal_lane_ids=sig_ids,
        signal_states=sig_states,
        signal_points=sig_points,
    )


def iter_scenarios(path: str):
    for payload in P.iter_tfrecords(path):
        yield parse_scenario(payload)
