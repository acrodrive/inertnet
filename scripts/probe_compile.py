"""Short training-loop probe: does torch.compile still leak live CUDA memory?

torch 2.4.1 leaked ~0.9 MB/step of unreclaimable live CUDA memory under
`torch.compile` (inductor bug) and OOM'd the first real run at ~step 5400.
eager did not. This runs a few hundred real training steps and reports the
slope of `torch.cuda.memory_allocated()` vs step so a long run's fate can be
predicted in minutes instead of hours. Also measures it/s and peak memory so
compile-vs-eager and batch-size headroom on the 5090 can be compared.

    python scripts/probe_compile.py --compile --batch-size 8 --steps 400
    python scripts/probe_compile.py --no-compile --batch-size 12 --steps 300

Writes nothing to runs/; prints a summary table at the end.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from mambahybrid.config import Config
from mambahybrid.data import WOMDDataset, collate
from mambahybrid.losses import compute_losses
from mambahybrid.model import MambaHybrid
from mambahybrid.train import lr_lambda, move


def linfit_slope(xs, ys):
    """Least-squares slope of ys vs xs (both plain lists)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--compile", dest="compile", action="store_true")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.set_defaults(compile=True)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=60,
                    help="steps to skip before fitting the leak slope "
                         "(compile trace + allocator warmup)")
    args = ap.parse_args()

    cfg = Config.load(args.config) if os.path.exists(args.config) else Config()
    cfg.compile = args.compile
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    assert torch.cuda.is_available(), "no CUDA device"
    device = "cuda"
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"compile={cfg.compile}  batch_size={cfg.batch_size}  "
          f"grad_checkpoint={cfg.grad_checkpoint}  amp={cfg.amp and not cfg.compile}")

    pc = cfg.preprocess()
    train_ds = WOMDDataset(cfg.train_shards, pc, "train", cfg.cache_dir, cfg.seed)
    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, collate_fn=collate,
        persistent_workers=cfg.num_workers > 0, pin_memory=True,
    )

    model = MambaHybrid(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, cfg))
    use_amp = cfg.amp and not cfg.compile
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()

    MB = 1024 * 1024
    samples = []          # (step, allocated_MB) after the step, post-sync
    data_iter = iter(loader)
    t0 = time.time()
    tw = None             # wall clock at end of warmup, for clean it/s

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = move(batch, device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(batch)
            losses = compute_losses(out, batch, cfg)
        opt.zero_grad(set_to_none=True)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()

        del out, losses, batch

        if step == args.warmup:
            torch.cuda.synchronize()
            tw = time.time()

        if step % args.log_every == 0 or step == args.steps:
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / MB
            reserved = torch.cuda.memory_reserved() / MB
            peak = torch.cuda.max_memory_allocated() / MB
            its = step / (time.time() - t0)
            print(f"step {step:>4} | alloc {alloc:8.1f} MB | reserved {reserved:8.1f} MB "
                  f"| peak {peak:8.1f} MB | {its:.2f} it/s")
            if step >= args.warmup:
                samples.append((step, alloc))

    print("\n" + "=" * 64)
    if len(samples) >= 2:
        xs = [s for s, _ in samples]
        ys = [a for _, a in samples]
        slope = linfit_slope(xs, ys)          # MB per step
        drift = slope * (xs[-1] - xs[0])
        print(f"allocated slope (post-warmup): {slope*1024:+.1f} KB/step  "
              f"({slope:+.4f} MB/step)")
        print(f"  drift over {xs[0]}->{xs[-1]}: {drift:+.1f} MB")
        proj = slope * 22000
        print(f"  projected over a 22k-step run: {proj:+.0f} MB "
              f"({'LEAK — will OOM' if proj > 4000 else 'ok' if abs(proj) < 1500 else 'watch'})")
    if tw is not None:
        clean_its = (args.steps - args.warmup) / (time.time() - tw)
        print(f"clean it/s (after step {args.warmup}): {clean_its:.3f}")
    print(f"peak allocated: {torch.cuda.max_memory_allocated()/MB:.0f} MB / "
          f"{torch.cuda.get_device_properties(0).total_memory/MB:.0f} MB total")
    print("=" * 64)


if __name__ == "__main__":
    main()
