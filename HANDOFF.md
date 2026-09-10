# Session handoff — 2026-09-10 (end of day)

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
- Recipe (exp5, `configs/exp5.yaml`): compile on, d_model 256, batch 4 (30.5 GB
  peak), lr 2e-4, EWTA, agent-centric decoding. ~1.15 it/s, 35k steps ≈ 8.5 h.
  (exp4 recipe was d_model 128, batch 8, lr 1e-4, ~1.65 it/s.)

## The story so far (runs archived on disk: runs/expN_*_{0909,0910}/)

| run | change | final all/minADE | minFDE | MR |
|-----|--------|------|--------|-----|
| exp1 | lr 3e-4, lambda_traj 1 | **diverged** at ~step 3300 (aux tasks saturate, near-peak lr) | | |
| exp2 | lr 1e-4, grad_clip 0.5, lambda_traj 5, lambda_mode 1, traj smooth_l1 beta 0.1 | 2.53 | 5.57 | 0.808 |
| exp3 | agent-centric trajectory decoding (head predicts offsets in each slot's local frame, rotates back to scene) | 2.06 | 5.43 | 0.769 |
| exp4 | EWTA (top-M WTA, M annealed 6→1) — revives dead modes | **1.87** (best 1.82 @ step 12000) | **4.10** | 0.764 |
| exp5 | scale up: d_model 128→256, mamba 4→6, decoder 2→4, heads 4→8 (1.94→12.0 M), lr 1e-4→2e-4 (swept), 17k→35k steps | **1.34** (best **1.25** @ step 26000) | **3.02** | 0.564 |

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

Diagnostics on exp5 best.pt (step 26000, `runs/exp5_scale_0910/`):
- minADE **1.29**, minFDE 3.08, MR 0.59 — a 31% cut over exp4, roughly the
  HANDOFF "expected ~1.4". Beats CV (4.47) comfortably.
- **Occlusion gap narrowed**: occluded minADE 1.75 vs clean 1.25 (~0.5 m, was
  ~0.9 m at exp4). Still positive — the research signal survives — but capacity
  closes half of it. Occluded n=125 focal agents, so this slice is noisy.
- Modes still healthy: effective 5.70/6, winner share `[.19 .19 .23 .18 .13 .08]`,
  no dead modes. Gain from K: minADE_1 2.77 → minADE_K 1.29 (53% lower).
- Not mean-regressing: endpoint displacement 31.8 m ≈ GT 31.9 m.
- ADE by horizon: 0.42 / 0.79 / 1.30 / 2.09 / **3.37** m at 1..5 s — same shape
  as exp4 (long horizon is the ceiling), shifted down ~1 m everywhere.
- **lr 2e-4 ran hot in the second half**: val bounced 1.25 (26k) → 1.51 (28k) →
  1.34 (35k). best.pt (26k) is a real ~0.09 better than the final. For the
  ablation, prefer lr 1e-4 (sweep had it tied with 2e-4 at 6k, and it should
  converge smoother over a full run) or keep 2e-4 and select on best.pt.

## Why exp4 plateaued (~1.8) — mostly capacity, confirmed by exp5

Train error ≈ val error ≈ ~1.8 m at exp4 → **underfitting**. exp5 tested cause 1
and it was most of the story:
1. **Capacity** (confirmed): 1.94 M → 12.0 M (d_model 128→256) bought minADE
   1.87 → 1.29 with no other structural change. Still underfitting-ish (train ≈
   val ≈ ~1.3), and val is now noisy run-to-run, so returns on pure scale are
   thinning on this data budget (~44k scenarios).
2. **Architecture bottleneck** (untested, → exp6): TrajectoryHead is a 3-layer
   MLP → `Linear(d→K·50·2)` producing all K×50×2 from ONE pooled d-vector. No
   per-mode structure, no temporal structure, and the map is consulted ONCE
   (MapDecoder makes f_upd_t) then compressed — the head can't attend to specific
   lanes while generating the far future. Long-horizon ADE (3.4 m @ 5 s) and
   MR 0.56 are where this would pay off.

## Next steps (in order)

### exp5 — scale up — DONE (2026-09-10). minADE 1.87 → 1.25 (best.pt @ 26k).
- Final recipe: `configs/exp5.yaml` — d_model 256, mamba_layers 6,
  decoder_layers 4, n_heads 8, **batch 4** (bs 8/6 OOM at d_model 256; bs 4 =
  30.5/32 GB peak, flat), lr **2e-4** (swept {5e-5, 1e-4, 2e-4}, all landed
  ~2.0-2.1 at 6k; 2e-4 fastest, no divergence), max_steps 35000, compile on.
- **1.15 it/s at d_model 256** (no fused kernels needed) — 35k steps ≈ 8.5 h.
  torchinductor cache carries between runs, so restarts skip the compile trace.
- Sweep runs archived: `runs/exp5_lr{5e-5,1e-4,2e-4}_0910/`. Full run:
  `runs/exp5_scale_0910/` (best.pt = step 26000, last.pt = step 35000 + opt).
- Sweep/driver helpers added: `scripts/make_sweep_configs.py`,
  `scripts/pick_sweep_winner.py`, `scripts/run_exp5_sweep.sh`.
- **Open issue**: second-half val instability at lr 2e-4 (see diagnostics above).
  Decide before the ablation — lr 1e-4 for the full ablation runs is the safe call.

### exp6 — query-based trajectory decoder (real architecture change)
- Replace MLP head with K mode queries doing cross-attention to map polylines +
  agents over ~3 layers (MTR/Wayformer style). Map is queried *during* generation.
- ~1.0-1.2 it/s expected (~4 h). Does NOT need more data (query+attn is
  param-efficient; map-conditioning aids generalization). Compile still works.
- To cut cost: decode from every Nth eligible step instead of all ~70.
- Expected: ~1.1.

### Then: the ablation (the actual deliverable)
- Lock the mamba recipe. `configs/exp5.yaml` is the current candidate — consider
  dropping lr to 1e-4 for stability, otherwise it is ready.
- Update `configs/ablation_gru.yaml` / `ablation_attn.yaml` to match it exactly
  (they're stale: batch 2/4, max_steps 120k, old lr, no compile/EWTA). Match via
  `mamba_layers` (GRUEncoder/CausalAttnEncoder both read `cfg.mamba_layers` for
  their layer count) and report the resulting param counts.
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
mkdir -p runs/exp6
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python -m mambahybrid.train --config configs/exp5.yaml > runs/exp6/train.log 2>&1 &
tail -f runs/exp6/train.log     # first ~50 steps are the compile trace (warm cache: faster)
```
- Fresh pod: `pip install -r requirements.txt && pip install --no-deps blackboxprotobuf==1.0.1 && pip install --no-deps -e .`
- lr sweep: `python scripts/make_sweep_configs.py --base <cfg> --lrs 5e-5 1e-4 2e-4 --batch-size 4 --steps 6000 --tag <tag>` then `bash` a loop over the generated `configs/<tag>_lr*.yaml`; summarise with `scripts/pick_sweep_winner.py runs/<tag>_lr*`.
- Resume: `--resume runs/<dir>/last.pt` (saved every val_every with opt state).
- Diagnose: `python scripts/diagnose_ckpt.py --ckpt runs/<dir>/best.pt --config configs/exp5.yaml`
- Compile/mem/leak probe: `python scripts/probe_compile.py --compile --batch-size N`
- **Kill training**: `pkill -9 -f mambahybrid.train` then check `nvidia-smi`
  (main proc sometimes lingers on DataLoader worker teardown holding ~23 GB).

## State

- exp5 committed. `configs/exp5.yaml` holds the locked scale recipe (lr 2e-4;
  see the "open issue" note about lr 1e-4).
- `configs/default.yaml` is unchanged (still the exp4 d_model-128 recipe) — use
  `configs/exp5.yaml` as the base from here on.
- Runs are gitignored; archived on disk: exp1-4 in `runs/expN_*_0909/`, exp5 in
  `runs/exp5_*_0910/`.
- Env: RTX 5090, torch 2.8.0+cu128, fresh pods need the pip install line above.
- Nothing running. GPU free.
- **Next**: exp6 (query decoder) per the roadmap, or skip straight to the
  ablation with `configs/exp5.yaml` locked (the portfolio deliverable).
