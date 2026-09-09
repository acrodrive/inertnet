# Session handoff — 2026-09-09 (end of day)

Working notes so tomorrow's session picks up cleanly. Delete when stale.

## Goal (important context)

This project is a **grad-school application portfolio piece**, not a SOTA attempt.
What matters: a clean research narrative (occlusion-robust prediction via a
selective-SSM state model), a **controlled ablation** (mamba vs gru vs attn for
maintaining state through occlusion), honest evaluation, and the systematic
debugging story. Absolute minADE ~1.5 with a convincing ablation beats chasing
0.6 (which needs 10-50x params + 10x data + weeks — re-implementing MTR).

WOMD number intuition: minADE is metres of avg error over the 5 s future (best of
K=6 modes). ~2 m ≈ half a lane off, can't tell the lane at 5 s. ~1 m ≈ usable.
~0.6 m ≈ SOTA. MissRate = fraction where all 6 modes miss the 5 s endpoint by >2 m.

## Environment

- RTX 5090 32 GB, torch 2.8.0+cu128. `torch.compile` leak (2.4.1) is FIXED — verified.
- Data: `/workspace/datasets/waymo/motion/` — training/ 90 shards (~44k scenarios,
  ~10% of WOMD), validation/ 10 shards. No cache (disk-limited).
- Recipe: compile on, batch 8 (22 GB peak), lr 1e-4, EWTA, agent-centric decoding.
  ~1.65 it/s, 17k steps ≈ 2.8 h.

## The story so far (all runs archived in runs/expN_*_0909/)

| run | change | final all/minADE | minFDE | MR |
|-----|--------|------|--------|-----|
| exp1 | lr 3e-4, lambda_traj 1 | **diverged** at ~step 3300 (aux tasks saturate, near-peak lr) | | |
| exp2 | lr 1e-4, grad_clip 0.5, lambda_traj 5, lambda_mode 1, traj smooth_l1 beta 0.1 | 2.53 | 5.57 | 0.808 |
| exp3 | agent-centric trajectory decoding (head predicts offsets in each slot's local frame, rotates back to scene) | 2.06 | 5.43 | 0.769 |
| exp4 | EWTA (top-M WTA, M annealed 6→1) — revives dead modes | **1.87** (best 1.82 @ step 12000) | **4.10** | 0.764 |

Diagnostics (`scripts/diagnose_ckpt.py`) on exp4 best.pt (step 12000):
- **Mode collapse FIXED**: effective modes 2.06/6 (exp3) → **4.91/6** (exp4). WTA
  winner share now `[.26 .23 .17 .24 .06 .04]` (was `[.57 0 0 0 0 .43]`).
  Gain from K modes: minADE_1 3.39 → minADE_K 1.81 (47% lower).
- Beats constant-velocity (CV minADE 4.4, model 1.8). Not mean-regression
  (endpoint displacement model 30.4 m ≈ GT 30.9 m).
- ADE by horizon: 0.80 / 1.18 / 1.82 / 2.77 / **4.30** m at 1..5 s — long horizon
  is where it breaks; near-term is fine.
- **Occlusion gap persists**: occluded minADE 2.63 vs clean 1.74 (~0.9 m). This is
  the key research signal — measure it across the ablation.

## Why it plateaus (~1.8, not a tuning artifact)

Train error ≈ val error ≈ ~1.8 m → **underfitting**, not overfitting, not lr.
Two causes:
1. **Capacity**: 1.94 M params, d_model 128. SOTA is 10-65 M. → exp5.
2. **Architecture bottleneck**: TrajectoryHead is a 3-layer MLP → `Linear(128→600)`
   producing all K×50×2 from ONE pooled 128-d vector. No per-mode structure, no
   temporal structure, and the map is consulted ONCE (MapDecoder makes f_upd_t)
   then compressed — the head can't attend to specific lanes while generating the
   far future. → exp6.

## Next steps (in order)

### exp5 — scale up (config + hyperparams, little new code)
- `d_model` 128 → 256, `mamba_layers` 4 → 6, `decoder_layers` 2 → 4, maybe
  `n_heads` 4 → 8. ~1.9 M → ~15 M params.
- `max_steps` → ~35000 (exp4 was still improving; also more params).
- Watch: the 91-step Python loop slows at d_model 256. If it drops below ~0.9
  it/s, install fused mamba kernels — `CAUSAL_CONV1D_FORCE_BUILD=TRUE
  MAMBA_FORCE_BUILD=TRUE pip install "causal-conv1d>=1.5.0" "mamba-ssm>=2.2.4"`
  (compiles from source on Blackwell, ~few min; Dockerfile has the line commented).
- May need batch 8 → 6 for memory.
- **Also do a small lr sweep here** {5e-5, 1e-4, 2e-4} — 1e-4 was picked to stop
  exp1's divergence, never properly tuned for the current (more stable) recipe.
- Expected: ~1.4.

### exp6 — query-based trajectory decoder (real architecture change)
- Replace MLP head with K mode queries doing cross-attention to map polylines +
  agents over ~3 layers (MTR/Wayformer style). Map is queried *during* generation.
- ~1.0-1.2 it/s expected (~4 h). Does NOT need more data (query+attn is
  param-efficient; map-conditioning aids generalization). Compile still works.
- To cut cost: decode from every Nth eligible step instead of all ~70.
- Expected: ~1.1.

### Then: the ablation (the actual deliverable)
- Lock the mamba recipe (after exp5, or exp6 if done).
- Update `configs/ablation_gru.yaml` / `ablation_attn.yaml` to match it exactly
  (they're stale: batch 2/4, max_steps 120k, old lr, no compile/EWTA).
- Per-encoder: short lr sweep {5e-5, 1e-4, 2e-4} (~6k steps), pick best, full run.
  Match param count via layer count; report it.
- `attn` can't use grad_checkpoint/compile (state grows per step) — document the
  fairness caveat, match what you can (params, data, steps, lr).
- Report all slices, emphasise **occluded** minADE/MR. Table:
  `runs/ablation_{mamba,gru,attn}/`.
- Optional: 3 seeds for the headline comparison.

### Portfolio polish
- Qualitative viz: predicted K trajectories vs GT on a few scenes, showing
  occlusion recovery. (No script yet — `scripts/` would be the place.)
- Write-up: hypothesis → system → ablation → honest limitations + "what more
  compute would buy".

## How to run

```bash
cd /workspace/inertnet
# edit configs/default.yaml (ckpt_dir, and exp5 scale params) first
mkdir -p runs/exp5
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python -m mambahybrid.train --config configs/default.yaml > runs/exp5/train.log 2>&1 &
tail -f runs/exp5/train.log     # first ~80 steps are the compile trace
```
- Resume: `--resume runs/exp5/last.pt` (saved every val_every=1500 with opt state).
- Diagnose: `python scripts/diagnose_ckpt.py --ckpt runs/exp5/best.pt --config runs/exp5/config.yaml`
- Compile/mem/leak probe: `python scripts/probe_compile.py --compile --batch-size N`
- **Kill training**: `pkill -9 -f mambahybrid.train` then check `nvidia-smi`
  (main proc sometimes lingers on DataLoader worker teardown holding ~23 GB).

## State

- All code committed (last: `59dd549 losses: EWTA trajectory assignment (exp4)`).
- `configs/default.yaml` `ckpt_dir` still says `runs/exp4` — change before next run.
- Runs are gitignored; archived on disk in `runs/expN_*_0909/`.
- Nothing running. GPU free.
