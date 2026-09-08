# Session handoff — 2026-09-08

Working notes so a fresh Claude Code session (or future me) can pick up. Delete when stale.

## What was done this session

1. **Dataset paths fixed.** Data lives at `/workspace/datasets/waymo/motion/{training,validation,testing}/`.
   `training/` = 90 shards (00000–00089), `validation/` = 10 shards (full 91-step + future GT),
   `testing/` = history-only (unusable for val). Configs now point at `training/` for train,
   `validation/` for val. `scripts/inspect_shard.py` hardened against empty globs.

2. **Parser hang fixed.** `parse_scenario` now passes an explicit `SCENARIO_TYPEDEF`
   (`src/mambahybrid/data/womd_proto.py`) to blackboxprotobuf. Without it, some scenarios sent
   the type-guesser into minutes-long backtracking. Byte-identical output on fast records;
   full-shard sweeps clean.

3. **torch.compile tried and disabled.** ~2.3x on the step loop but leaks ~0.9 MB/step of live
   CUDA memory on torch 2.4.1 (inductor bug — not the checkpoint wrapper, not autocast, not
   dynamic shapes; `torch.compiler.reset()` doesn't reclaim it). OOM'd the first real run at
   ~step 5400. Wiring kept in `model/hybrid.py`; `config.compile = False`.

4. **Running eager.** `configs/default.yaml`: `compile: false`, `amp: false` (no speedup here,
   scan is fp32-only), `batch_size: 6` (~17.4 GB, no drift), `max_steps: 22000` (~3 epochs,
   ~21 h at ~0.30 it/s), `val_every: 1500`.

## State

- All changes committed. Last commit: `train: disable torch.compile (leaks on 2.4.1), run eager`.
- **Not yet run.** No `runs/exp1/` (the crashed compile run is archived at
  `runs/exp1_crashed_compile_*/`).
- First real run's val (before it crashed): minADE 20.7 → 12.5 → 10.9 → ~6.75 by step 2000.
  Model is learning (not degenerate) but far from a good number (~0.7). Needs more training
  and/or LR tuning — that's open research, not a bug.

## To start training

```bash
cd /workspace/inertnet
mkdir -p runs/exp1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python -m mambahybrid.train --config configs/default.yaml > runs/exp1/train.log 2>&1 &
```
Monitor: `tail -f runs/exp1/train.log`. First val at step 1500 (~1.5 h).
Extend later: add `--resume runs/exp1/last.pt` (saves every `val_every` steps with opt state).

## Speed options not yet taken

- Install `mamba_ssm` + `causal_conv1d` (fused kernels; the 91-step Python loop stays, so
  gains are limited).
- Process-restart wrapper around `compile` (restart every ~4000 steps from `last.pt` to shed
  the leak) — keeps ~2.2x. Needs a `--steps-this-run` arg in `train.py` + a shell loop.
- torch upgrade (2.5+) may fix the compile leak — retest before trusting.
