"""Drive the full ablation (spec 3.4.2) end to end, unattended, resumable.

For gru and attn: short lr sweep (6k steps, {5e-5, 1e-4, 2e-4}) -> pick the
winner by val all/minADE -> full 20k-step run at that lr. For mamba: no
sweep (exp5 already ran one; lr 2e-4 was its winner) -> straight to a fresh
20k-step run. All three use max_steps=20000 so the warmup/cosine-decay
schedule shape is identical across arms (see configs/ablation_gru.yaml
header / HANDOFF.md 2026-09-11) — reusing exp5's step-26000 checkpoint
(picked mid-schedule out of a 35k run) would leave it under-annealed
relative to a schedule that actually ends at the comparison point.

Resumable: re-running this script skips any training run whose log.jsonl
already reaches its target step (so a crash partway through only re-does the
one run that was interrupted, not the whole pipeline). Safe to `nohup` for
the ~30h total and reattach/rerun later.

    nohup python -u scripts/run_ablation.py > runs/ablation_driver.log 2>&1 &
    tail -f runs/ablation_driver.log

At the end, runs scripts/diagnose_ckpt.py on each arm's best.pt and writes
runs/ablation_report.txt with all three side by side.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mambahybrid.config import Config  # noqa: E402

SWEEP_STEPS = 6000
FULL_STEPS = 20000
SWEEP_LRS = [5e-5, 1e-4, 2e-4]


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def lr_suffix(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")


def last_step(ckpt_dir: str) -> int:
    p = ROOT / ckpt_dir / "log.jsonl"
    if not p.exists():
        return -1
    last = -1
    with open(p) as fh:
        for line in fh:
            try:
                last = max(last, json.loads(line).get("step", -1))
            except json.JSONDecodeError:
                continue
    return last


def best_val_minade(ckpt_dir: str) -> float:
    p = ROOT / ckpt_dir / "log.jsonl"
    best = float("inf")
    if not p.exists():
        return best
    with open(p) as fh:
        for line in fh:
            d = json.loads(line)
            if "val" in d:
                best = min(best, d["val"]["all/minADE"])
    return best


def run_training(cfg_path: Path, ckpt_dir: str, target_step: int) -> None:
    if last_step(ckpt_dir) >= target_step:
        log(f"skip (already done): {ckpt_dir} (log shows step {last_step(ckpt_dir)})")
        return
    log(f"RUN: python -u -m mambahybrid.train --config {cfg_path}")
    subprocess.run(
        [sys.executable, "-u", "-m", "mambahybrid.train", "--config", str(cfg_path)],
        check=True, cwd=ROOT,
    )


def sweep_and_pick(tag: str, base_cfg: str, batch_size: int) -> float:
    results = {}
    for lr in SWEEP_LRS:
        suf = lr_suffix(lr)
        cfg_path = ROOT / f"configs/{tag}_lr{suf}.yaml"
        ckpt_dir = f"runs/{tag}_lr{suf}"
        if not cfg_path.exists():
            cfg = Config.load(base_cfg)
            cfg.lr = lr
            cfg.batch_size = batch_size
            cfg.max_steps = SWEEP_STEPS
            cfg.val_every = 1000
            cfg.warmup_iters = min(cfg.warmup_iters, SWEEP_STEPS // 6)
            cfg.ckpt_dir = ckpt_dir
            cfg.dump(str(cfg_path))
        run_training(cfg_path, ckpt_dir, SWEEP_STEPS)
        results[lr] = best_val_minade(ckpt_dir)
    winner = min(results, key=results.get)
    log(f"sweep {tag}: {results} -> winner lr={winner}")
    return winner


def full_run(tag: str, base_cfg: str, lr: float, batch_size: int) -> str:
    ckpt_dir = f"runs/{tag}"
    cfg_path = ROOT / f"configs/{tag}_final.yaml"
    cfg = Config.load(base_cfg)
    cfg.lr = lr
    cfg.batch_size = batch_size
    cfg.max_steps = FULL_STEPS
    cfg.val_every = 2000
    cfg.ckpt_dir = ckpt_dir
    cfg.dump(str(cfg_path))
    run_training(cfg_path, ckpt_dir, FULL_STEPS)
    return str(cfg_path)


def diagnose(tag: str, cfg_path: str) -> str:
    ckpt = ROOT / f"runs/{tag}" / "best.pt"
    if not ckpt.exists():
        return f"{tag}: no best.pt found, skipping diagnostics"
    out = subprocess.run(
        [sys.executable, "scripts/diagnose_ckpt.py", "--ckpt", str(ckpt), "--config", cfg_path],
        check=True, cwd=ROOT, capture_output=True, text=True,
    )
    return f"=== {tag} ({ckpt}) ===\n{out.stdout}"


def main() -> None:
    log("=== ablation driver start ===")

    gru_lr = sweep_and_pick("ablation_gru_sweep", "configs/ablation_gru.yaml", batch_size=4)
    gru_cfg = full_run("ablation_gru", "configs/ablation_gru.yaml", gru_lr, batch_size=4)

    attn_lr = sweep_and_pick("ablation_attn_sweep", "configs/ablation_attn.yaml", batch_size=2)
    attn_cfg = full_run("ablation_attn", "configs/ablation_attn.yaml", attn_lr, batch_size=2)

    mamba_cfg = full_run("ablation_mamba", "configs/exp5.yaml", 2e-4, batch_size=4)

    log("=== all training done, running diagnostics ===")
    report = "\n\n".join([
        diagnose("ablation_mamba", mamba_cfg),
        diagnose("ablation_gru", gru_cfg),
        diagnose("ablation_attn", attn_cfg),
    ])
    (ROOT / "runs/ablation_report.txt").write_text(report)
    log("wrote runs/ablation_report.txt")
    log("=== ablation driver done ===")


if __name__ == "__main__":
    main()
