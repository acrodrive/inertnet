"""Quick stats over WOMD shard(s): occlusion rates, gap lengths, signal states.

    python scripts/inspect_shard.py "<glob>" [max_scenarios]

    # ~50 scenarios is plenty for a sanity check; each proto parse is ~seconds
    python scripts/inspect_shard.py \
        "/workspace/datasets/waymo/motion/training/training.tfrecord-0000*-of-01000" 50
"""
import collections
import glob
import sys

import numpy as np

sys.path.insert(0, "src")
from mambahybrid.data.parser import iter_scenarios  # noqa: E402


def main(pattern: str, max_scenarios: int = 50):
    files = sorted(glob.glob(pattern))
    print(f"{len(files)} shard(s)")
    if not files:
        print(f"no files match {pattern!r} -- check the path/glob")
        return
    n_sc = n_tr = n_any_inv = n_gap = n_ttp = n_ttp_gap = 0
    gap_lens = []
    state_hist = collections.Counter()
    for f in files:
        for sc in iter_scenarios(f):
            n_sc += 1
            ttp = set(int(i) for i in sc.tracks_to_predict)
            for a in range(sc.num_agents):
                v = sc.valid[a]
                if not v.any():
                    continue
                n_tr += 1
                if (~v).any():
                    n_any_inv += 1
                lo, hi = np.argmax(v), len(v) - 1 - np.argmax(v[::-1])
                seg = v[lo:hi + 1]
                has_gap = (~seg).any()
                if has_gap:
                    n_gap += 1
                    # gap run lengths
                    d = np.diff(np.concatenate([[1], seg.astype(int), [1]]))
                    starts = np.where(d == -1)[0]
                    ends = np.where(d == 1)[0]
                    gap_lens.extend((ends - starts).tolist())
                if a in ttp:
                    n_ttp += 1
                    n_ttp_gap += has_gap
            for t in range(sc.num_steps):
                for s in sc.signal_states[t]:
                    state_hist[int(s)] += 1
            if n_sc >= max_scenarios:
                break
        if n_sc >= max_scenarios:
            break

    print(f"scenarios: {n_sc}   tracks: {n_tr}")
    if n_tr == 0:
        print("no valid tracks parsed -- nothing to summarise")
        return
    print(f"tracks with any invalid step : {n_any_inv} ({100*n_any_inv/n_tr:.1f}%)")
    print(f"tracks with interior gap     : {n_gap} ({100*n_gap/n_tr:.1f}%)")
    if gap_lens:
        g = np.array(gap_lens)
        print(f"  gap length steps: median {np.median(g):.0f}  mean {g.mean():.1f}  max {g.max()}")
    print(f"focal (tracks_to_predict)    : {n_ttp}, with gap {n_ttp_gap} ({100*n_ttp_gap/max(1,n_ttp):.1f}%)")
    print(f"signal state histogram       : {dict(state_hist)}")


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else \
        "/workspace/datasets/waymo/motion/training/training.tfrecord-00000-of-01000"
    main(pat, int(sys.argv[2]) if len(sys.argv) > 2 else 50)
