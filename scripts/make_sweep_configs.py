"""Generate short lr-sweep configs from a base config.

    python scripts/make_sweep_configs.py --base configs/exp5.yaml \
        --lrs 5e-5 1e-4 2e-4 --batch-size 4 --steps 6000 --tag exp5

Writes configs/<tag>_lr<lr>.yaml, each pointed at runs/<tag>_lr<lr>/. The EWTA
M-anneal in losses.py keys off step/max_steps, so a short sweep compresses the
schedule identically across the three runs — fine for picking a relative winner.
"""
from __future__ import annotations

import argparse

from mambahybrid.config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lrs", nargs="+", type=float, required=True)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    for lr in args.lrs:
        cfg = Config.load(args.base)
        cfg.lr = lr
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        cfg.max_steps = args.steps
        cfg.val_every = args.val_every
        cfg.warmup_iters = min(cfg.warmup_iters, args.steps // 6)
        lr_s = f"{lr:.0e}".replace("e-0", "e-")
        cfg.ckpt_dir = f"runs/{args.tag}_lr{lr_s}"
        path = f"configs/{args.tag}_lr{lr_s}.yaml"
        cfg.dump(path)
        print(f"wrote {path}  (lr={lr}, steps={cfg.max_steps}, bs={cfg.batch_size}, "
              f"ckpt_dir={cfg.ckpt_dir})")


if __name__ == "__main__":
    main()
