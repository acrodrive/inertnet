"""Parser sanity against the local shard."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mambahybrid.data import womd_proto as P
from mambahybrid.data.parser import parse_scenario

SHARD = os.environ.get(
    "WOMD_TEST_SHARD",
    "/workspace/datasets/waymo/motion/training/training.tfrecord-00000-of-01000",
)


def test_offsets_match_stream():
    offs = P.record_offsets(SHARD)
    assert len(offs) > 400
    for k, payload in enumerate(P.iter_tfrecords(SHARD)):
        if k in (0, 1, len(offs) - 1):
            assert P.read_record_at(SHARD, offs[k]) == payload


def test_scenario_fields():
    sc = parse_scenario(P.read_record_at(SHARD, P.record_offsets(SHARD)[0]))
    assert sc.num_steps == 91
    assert sc.current_time_index == 10
    assert sc.num_agents > 10
    assert len(sc.polylines) > 50
    assert len(sc.signal_states) == 91
    # a focal agent has a mostly-complete track
    a = int(sc.tracks_to_predict[0])
    assert sc.valid[a].sum() > 40
    v = sc.valid[a]
    assert np.isfinite(sc.xyz[a, v]).all()
    # sizes are plausible metres
    sizes = sc.size_lwh[sc.valid]
    assert (sizes[:, 0] < 40).all() and (sizes[:, 0] >= 0).all()


if __name__ == "__main__":
    test_offsets_match_stream()
    test_scenario_fields()
    print("parser tests OK")
