"""Summarise the exp5 lr sweep: print each run's val-metric curve and the best
all/minADE, then name the winner.

    python scripts/pick_sweep_winner.py runs/exp5_lr5e-5 runs/exp5_lr1e-4 runs/exp5_lr2e-4
"""
from __future__ import annotations

import json
import sys


def load_vals(run_dir: str):
    rows = []
    try:
        with open(f"{run_dir}/log.jsonl") as fh:
            for line in fh:
                d = json.loads(line)
                if "val" in d:
                    rows.append(d["val"])
    except FileNotFoundError:
        pass
    return rows


def main():
    runs = sys.argv[1:]
    best = {}
    for r in runs:
        vals = load_vals(r)
        if not vals:
            print(f"{r}: no val rows yet")
            continue
        print(f"\n{r}")
        for v in vals:
            print(f"  step {v['step']:>6}  all/minADE {v['all/minADE']:.3f}  "
                  f"minFDE {v['all/minFDE']:.3f}  MR {v['all/MissRate']:.3f}  "
                  f"occ/minADE {v.get('occluded/minADE', float('nan')):.3f}")
        b = min(v["all/minADE"] for v in vals)
        best[r] = b
        print(f"  -> best all/minADE {b:.3f}")
    if best:
        win = min(best, key=best.get)
        print(f"\nwinner: {win}  (all/minADE {best[win]:.3f})")
        print("best per run:", {k: round(v, 3) for k, v in best.items()})


if __name__ == "__main__":
    main()
