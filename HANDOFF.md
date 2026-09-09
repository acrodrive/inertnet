# Session handoff — 2026-09-09

Working notes so a fresh Claude Code session (or future me) can pick up. Delete when stale.

## What was done (2026-09-08)

1. **Dataset paths fixed.** Data lives at `/workspace/datasets/waymo/motion/{training,validation,testing}/`.
   `training/` = 90 shards (00000–00089), `validation/` = 10 shards (full 91-step + future GT),
   `testing/` = history-only (unusable for val). Configs now point at `training/` for train,
   `validation/` for val. `scripts/inspect_shard.py` hardened against empty globs.

2. **Parser hang fixed.** `parse_scenario` now passes an explicit `SCENARIO_TYPEDEF`
   (`src/mambahybrid/data/womd_proto.py`) to blackboxprotobuf. Without it, some scenarios sent
   the type-guesser into minutes-long backtracking. Byte-identical output on fast records;
   full-shard sweeps clean.

## What was done (2026-09-09)

3. **Moved to RTX 5090 (32 GB, Blackwell sm_120), torch 2.8.0 + cu128.**

4. **torch.compile leak retested — FIXED on torch 2.8.** The 2.4.1 inductor bug leaked
   ~0.9 MB/step of live CUDA memory (OOM'd the first run at ~step 5400). On torch 2.8
   `torch.cuda.memory_allocated()` is dead flat over 400 steps at every batch size tried.
   `scripts/probe_compile.py` is the harness (short training-loop probe: leak slope, it/s,
   peak VRAM). Sweep on the 5090:

   | config              | peak VRAM | leak      | clean it/s | samples/s |
   |---------------------|-----------|-----------|------------|-----------|
   | eager,   batch 6    | 16.8 GB   | 0         | 0.94       | 5.6       |
   | compile, batch 6    | 16.7 GB   | 0         | 1.99       | 11.9      |
   | compile, batch 8    | 22.1 GB   | 0         | 1.71       | 13.7      |
   | compile, batch 10   | 27.7 GB   | 0         | 1.54       | 15.4      |
   | compile, batch 12   | OOM       | —         | —          | —         |

   VRAM grows ~linearly with batch (inductor allocates `B*300`-slot buffers).

5. **`configs/default.yaml` updated for the 5090:** `compile: true`, `batch_size: 8`,
   `lr: 3.0e-4` (unchanged — fine at batch 8), `max_steps: 17000` (~3 epochs at
   ~5500 steps/epoch), `val_every: 1500`. `amp: false` (train.py forces it off under
   compile anyway). Stale "leaks on 2.4.1" comments in `config.py` / `model/hybrid.py` /
   `Dockerfile` updated.

## State

- **Not yet run.** No `runs/exp1/` (the crashed compile run is archived at
  `runs/exp1_crashed_compile_0908/`).
- First real run's val (before it crashed, on 4090): minADE 20.7 → 12.5 → 10.9 → ~6.75 by
  step 2000. Model is learning (not degenerate) but far from a good number (~0.7). Needs
  more training and/or LR tuning — that's open research, not a bug.
- Expected wall clock now: ~2.7 h for 17k steps at ~1.7 it/s (compile, batch 8).

## To start training

```bash
cd /workspace/inertnet
mkdir -p runs/exp1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python -m mambahybrid.train --config configs/default.yaml > runs/exp1/train.log 2>&1 &
```
Monitor: `tail -f runs/exp1/train.log`. First ~80 steps are the compile trace (slow),
then ~1.7 it/s. First val at step 1500 (~15 min in).
Extend later: add `--resume runs/exp1/last.pt` (saves every `val_every` steps with opt state).

## Speed options not yet taken

- batch 10 (compile) is ~12% more samples/s than batch 8 but 27.7 GB peak — usable with
  `expandable_segments` if a longer run wants it, no headroom for anything else.
- Install `mamba_ssm` + `causal_conv1d` (fused kernels; the 91-step Python loop stays, so
  gains are limited). No prebuilt Blackwell wheels — compiles from source (see Dockerfile).
