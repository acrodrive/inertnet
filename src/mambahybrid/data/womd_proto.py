"""Raw Waymo Open Motion Dataset (Scenario proto) field map + decode helpers.

We parse the TFRecord/protobuf by hand (via blackboxprotobuf) so the data
pipeline has no TensorFlow / waymo-open-dataset dependency. Field numbers below
are from `waymo_open_dataset/protos/scenario.proto` and `.../map.proto`.

Verified against training.tfrecord-00000-of-01000 (2026-09).
"""
from __future__ import annotations

import struct

# ---------------------------------------------------------------- Scenario -----
# message Scenario
SC_SCENARIO_ID = "5"          # bytes (hex string)
SC_TIMESTAMPS = "1"           # repeated double, len 91
SC_CURRENT_TIME_INDEX = "10"  # int32, == 10
SC_TRACKS = "2"               # repeated Track
SC_DYNAMIC_MAP_STATES = "7"   # repeated DynamicMapState, len == len(timestamps)
SC_MAP_FEATURES = "8"         # repeated MapFeature
SC_SDC_TRACK_INDEX = "6"      # int32
SC_OBJECTS_OF_INTEREST = "4"  # repeated int32
SC_TRACKS_TO_PREDICT = "11"   # repeated RequiredPrediction

# message Track
TR_ID = "1"           # int32
TR_OBJECT_TYPE = "2"  # enum: 0 UNSET, 1 VEHICLE, 2 PEDESTRIAN, 3 CYCLIST, 4 OTHER
TR_STATES = "3"       # repeated ObjectState, len 91

# message ObjectState  (center_* are double / fixed64, the rest float / fixed32)
OS_CENTER_X = "2"
OS_CENTER_Y = "3"
OS_CENTER_Z = "4"
OS_LENGTH = "5"
OS_WIDTH = "6"
OS_HEIGHT = "7"
OS_HEADING = "8"
OS_VELOCITY_X = "9"
OS_VELOCITY_Y = "10"
OS_VALID = "11"  # bool

# message RequiredPrediction
RP_TRACK_INDEX = "1"
RP_DIFFICULTY = "2"

# message DynamicMapState -> repeated TrafficSignalLaneState lane_states = 1
DMS_LANE_STATES = "1"
# message TrafficSignalLaneState
TS_LANE = "1"        # int64  (id of the LaneCenter it controls)
TS_STATE = "2"       # enum, see LANE_STATE_* below
TS_STOP_POINT = "3"  # MapPoint

# TrafficSignalLaneState.State
LANE_STATE_UNKNOWN = 0
LANE_STATE_ARROW_STOP = 1
LANE_STATE_ARROW_CAUTION = 2
LANE_STATE_ARROW_GO = 3
LANE_STATE_STOP = 4
LANE_STATE_CAUTION = 5
LANE_STATE_GO = 6
LANE_STATE_FLASHING_STOP = 7
LANE_STATE_FLASHING_CAUTION = 8
N_LANE_STATES = 9

# ------------------------------------------------------------------- Map -------
# message MapFeature: id = 1, then a oneof:
MF_ID = "1"
MF_LANE = "3"        # LaneCenter
MF_ROAD_LINE = "4"   # RoadLine
MF_ROAD_EDGE = "5"   # RoadEdge
MF_STOP_SIGN = "7"   # StopSign
MF_CROSSWALK = "8"   # Crosswalk
MF_SPEED_BUMP = "9"  # SpeedBump
MF_DRIVEWAY = "10"   # Driveway

# message LaneCenter
LC_SPEED_LIMIT_MPH = "1"  # double
LC_TYPE = "2"             # 0 UNDEFINED, 1 FREEWAY, 2 SURFACE_STREET, 3 BIKE_LANE
LC_INTERPOLATING = "3"    # bool
LC_POLYLINE = "8"         # repeated MapPoint
LC_ENTRY_LANES = "9"      # packed int64
LC_EXIT_LANES = "10"      # packed int64

# message RoadLine / RoadEdge
RL_TYPE = "1"
RL_POLYLINE = "2"

# StopSign: lane = 1 (packed int64), position = 2 (MapPoint)
SS_LANE = "1"
SS_POSITION = "2"

# Crosswalk / SpeedBump / Driveway: polygon = 1 (repeated MapPoint)
POLY_POLYGON = "1"

# message MapPoint
MP_X = "1"
MP_Y = "2"
MP_Z = "3"

# Semantic layer ids we assign to each vectorized polyline token.
LAYER_LANE_FREEWAY = 0
LAYER_LANE_SURFACE = 1
LAYER_LANE_BIKE = 2
LAYER_ROAD_LINE = 3
LAYER_ROAD_EDGE = 4
LAYER_CROSSWALK = 5
LAYER_SPEED_BUMP = 6
LAYER_DRIVEWAY = 7
LAYER_STOP_SIGN = 8
N_MAP_LAYERS = 9

# Object type
OBJECT_TYPES = {0: "unset", 1: "vehicle", 2: "pedestrian", 3: "cyclist", 4: "other"}
N_OBJECT_CLASSES = 5

WOMD_HZ = 10
WOMD_DT = 0.1


# ---------------------------------------------------------------- decoders -----
def as_f64(bits) -> float:
    """fixed64 int bit-pattern -> double."""
    if isinstance(bits, float):
        return bits
    return struct.unpack("<d", struct.pack("<Q", int(bits) & 0xFFFFFFFFFFFFFFFF))[0]


def as_f32(bits) -> float:
    """fixed32 int bit-pattern -> float."""
    if isinstance(bits, float):
        return bits
    return struct.unpack("<f", struct.pack("<I", int(bits) & 0xFFFFFFFF))[0]


def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def point_xyz(mp: dict) -> tuple[float, float, float]:
    return (
        as_f64(mp.get(MP_X, 0.0)),
        as_f64(mp.get(MP_Y, 0.0)),
        as_f64(mp.get(MP_Z, 0.0)),
    )


def iter_tfrecords(path: str):
    """Yield raw protobuf payloads from a TFRecord file.

    TFRecord layout per record: uint64 length | uint32 crc | data | uint32 crc.
    CRCs are not checked.
    """
    with open(path, "rb") as fh:
        while True:
            header = fh.read(8)
            if len(header) < 8:
                return
            (length,) = struct.unpack("<Q", header)
            fh.read(4)  # length crc
            data = fh.read(length)
            fh.read(4)  # data crc
            if len(data) < length:
                return
            yield data


def record_offsets(path: str) -> list[int]:
    """Byte offset of every record payload, so shards can be read by index."""
    offsets: list[int] = []
    with open(path, "rb") as fh:
        while True:
            pos = fh.tell()
            header = fh.read(8)
            if len(header) < 8:
                return offsets
            (length,) = struct.unpack("<Q", header)
            offsets.append(pos + 12)          # skip 8-byte len + 4-byte len-crc
            fh.seek(length + 8, 1)            # 4-byte len-crc gap + payload + 4-byte data-crc


def read_record_at(path: str, payload_offset: int) -> bytes:
    with open(path, "rb") as fh:
        fh.seek(payload_offset - 12)
        (length,) = struct.unpack("<Q", fh.read(8))
        fh.read(4)
        return fh.read(length)
