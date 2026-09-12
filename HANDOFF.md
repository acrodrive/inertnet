# Session handoff — 2026-09-11 (updated 2026-09-12)

Working notes so the next session picks up cleanly. Delete when stale.

## 2026-09-12 update: pod was terminated mid-mamba-run, restarted clean

The pod running the ablation driver got killed externally (no error in the
log — loss was normal at step 4950, then the process just vanished; GPU was
idle, no process running when this session picked it back up). Status found:
**gru and attn had both already finished their full 20k runs** (confirmed via
`log.jsonl` max step = 20000 for both, `best.pt`/`last.pt` present). Only
**mamba's full run** was affected — it had reached **step 4950/20000 (~25%)**
before dying.

Two hazards found and avoided:
1. `run_training()` in `scripts/run_ablation.py` never passes `--resume` —
   simply rerunning the driver would have silently restarted mamba from
   scratch (fine) but with `log.jsonl` opened in append mode, so the old
   step-50..4950 records would stay mixed in before the new run's own
   step-50..20000 records. Handled by moving the old dir aside first:
   `runs/ablation_mamba_terminated_step4950_0912/` (kept for reference, not
   used by anything).
2. `train.py`'s own `--resume` path (`train.py:96-116`) restores `step` from
   the checkpoint but **not** the LR scheduler's internal counter (`sched` is
   constructed fresh before the resume block) — resuming would have replayed
   warmup+cosine-decay from scratch over the *remaining* steps instead of
   continuing the same 20k-step curve gru/attn used. This would have quietly
   broken the step-matched-schedule fairness protocol (see caveat further
   down). Not fixed (wasn't needed) — **flagging it here because it's a real
   latent bug** if anyone reaches for `--resume` on this codebase in the
   future; fix would be to also restore/fast-forward `sched`'s `last_epoch`
   (or just recompute lr via `lr_lambda(step, cfg)` and set it directly each
   iteration instead of relying on `LambdaLR`'s own counter).

**Decision made and executed**: restarted mamba's full run clean from step 0
(`runs/ablation_mamba/`), same config/lr/batch as before
(`configs/ablation_mamba_final.yaml`, lr 2e-4, batch 4, 20k steps). Verified
env first (fresh pod needed the same `blackboxprotobuf` +
`pip install --no-deps -e .` reinstall dance as the prior session — see "How
to run" below; torch 2.8.0+cu128 / 5090 was already fine this time, no
reinstall needed there). Driver relaunched via the same
`nohup python -u scripts/run_ablation.py > runs/ablation_driver.log 2>&1 &`
at 2026-09-12 12:38:43 — gru/attn stages skip instantly (already done),
mamba is training fresh (confirmed via GPU util 91%/32GB and process
running). Expected ~4.8h from relaunch for mamba alone, then the driver
auto-runs `diagnose_ckpt.py` on all three and writes
`runs/ablation_report.txt`. **Next session: just check
`tail -f runs/ablation_driver.log` and `runs/ablation_report.txt` — if the
driver ran to completion, the "After the driver finishes" section below is
the next step. If the pod died again mid-run, repeat this same restart
recipe (archive the partial `runs/ablation_mamba/` dir, relaunch the driver
— gru/attn will skip, mamba restarts clean).**

## ⚠️ Known methodology caveats (found via full-project code review, 2026-09-11)

Found late in this session, after the ablation was already ~24h in. **Decision:
let the running ablation finish as-is** — all five affect mamba/gru/attn
identically (same shared code path), so the ablation's core deliverable, the
**relative** mamba-vs-gru-vs-attn ranking (esp. the occlusion-gap comparison),
stays valid. What's compromised is the **absolute** numbers and one entire
evaluation axis. Must be disclosed in the write-up; #2 is cheap to fix without
retraining (see below).

1. **TrajectoryHead's coordinate anchor leaks ground truth**
   (`model/hybrid.py:131-133`). `traj_head(f_upd_t, batch["gt_pos"][:,:,t,:2],
   batch["gt_heading"][:,:,t])` — the local-frame offset prediction is rotated
   back to scene space using the *true, pre-occlusion* GT position/heading at
   step `t` as origin, not the model's own recovered estimate. When `t`
   (including eval's fixed `t0`) falls inside a synthetic-occlusion gap, the
   model is handed the exact answer to "where am I right now" for free before
   it even has to predict the future — the hard part of occlusion recovery is
   bypassed for the trajectory task specifically (GTHead's own reconstruction
   loss/metric, `recon_ade_ft`/`fupd`, is NOT affected — that's a separate,
   correctly-isolated head/metric). **Effect**: reported "occluded" minADE/FDE
   and the occluded-vs-clean gap are optimistically biased for all three arms
   equally. The relative ranking survives; "our occlusion gap is only 0.5 m"
   as an absolute claim does not, without this caveat.
   Real fix (needs retraining): anchor on the model's own `GTHead(f_upd_t)`
   position estimate instead of `batch["gt_pos"]`.
2. ~~**"occluded" vs "clean" val slice is a tautology equal to `synth_mask`**~~
   **FIXED 2026-09-12** — see "2026-09-12 update #2" below for the fix and
   the (significant) corrected result. Original text kept for context: the
   old `had_gap = synth_mask | (recon_mask & ~obj_valid)` algebraically
   reduced to exactly `synth_mask` (since `obj_valid == valid & ~synth_mask`
   and `synth_mask` already implies `valid`), so every natural-gap track
   (`gt_valid=False` mid-track, no synthetic gap injected) was silently
   bucketed as "clean" — spec 3.3.2/3.4.3's "자연 Occlusion 실검증" axis was
   not measured at all.
3. **L_recon trains on all originally-valid steps, not just synthetically-masked
   ones** (`losses.py:36`, `_recon_term` uses `batch["recon_mask"]` = `valid &
   slot_mask`, not intersected with `synth_mask`). Spec 3.2.6 says L_recon
   should fire "인위적으로 마스킹한 스텝에서만". Most of the recon loss mass
   comes from trivial, never-hidden steps, diluting the occlusion-recovery
   training signal (the reported eval metric `occlusion_reconstruction_error`
   is correctly isolated via `synth_mask` already — this is a training-signal
   dilution issue, not a reporting bug). Needs retraining to fix properly.
4. **Constant-velocity baseline (Baseline A) sees through occlusion**
   (`baselines.py:34`): reads `gt_valid`/`gt_pos` (pre-occlusion) directly
   instead of `obj_valid`/the post-occlusion input, so it always extrapolates
   from the true last observation even when the model's input was zeroed.
   Makes Baseline A stronger than it should be on occlusion-affected tracks —
   biases the comparison to be *more* conservative about the models' win
   over CV, not less. Low priority.
5. **`subtract_ego_velocity: false` in every shipped config** (all of exp1-5,
   ablation_*.yaml) — spec 3.1.1 calls for ego-relative vx/vy (subtract SDC's
   own t0 velocity); every run so far has trained on velocities that still
   carry the ego's own t0-frame velocity component. Same for all three arms,
   so relative comparison unaffected, but a real deviation from the spec's
   stated "pure relative motion" design intent. Needs retraining to fix.

## 2026-09-12 update #2: caveat #2 fixed — natural-occlusion slice now measured, and it changes the headline conclusion

Added `occlusion_gap_slices()` to `metrics.py`: `synth` (synth_mask fired, up
to t0), `natural` (a real `gt_valid=False` mid-track gap — invalid strictly
between the track's first and last valid step, so track start/end is not
miscounted as a gap — with no synthetic gap), `any` = `synth | natural`.
`trajectory_metrics()` now returns `synth_occluded`/`natural_occluded` slices
alongside the existing `all`/`occluded`/`clean`; `train.py`'s `evaluate()`
aggregates them too; `diagnose_ckpt.py`'s duplicate of the old buggy mask
logic was replaced with a call to the shared helper. Verified with a
synthetic-batch sanity check (leading/trailing invalid correctly NOT counted
as a gap, mid-track invalid correctly counted, synth vs natural correctly
disjoint) and `tests/test_pipeline.py` (updated for the new keys, 3/3 pass).
**No retraining needed** — re-ran `scripts/run_ablation.py` (all three
training stages skip instantly since already at their target step; only the
final `diagnose_ckpt.py` pass re-executes) to regenerate
`runs/ablation_report.txt` against the already-saved `best.pt` checkpoints.

**Corrected result (val split, n=1418 focal @ t0; 1248 clean / 125 synth /
45 natural — identical split and gap population across all three arms since
occlusion sampling is seeded independently of the model):**

| arm   | clean minADE | synth-occ minADE | natural-occ minADE | synth gap | natural gap |
|-------|-------------:|------------------:|--------------------:|----------:|------------:|
| mamba | 1.259 | 1.733 | 1.786 | +0.474 | +0.527 |
| gru   | 1.349 | 1.732 | 1.776 | +0.383 | +0.427 |
| attn  | 1.479 | 2.114 | 1.972 | +0.635 | +0.493 |

**This changes the ablation's interpretation.** mamba still wins on raw
`all/minADE` (1.323 vs gru 1.402 vs attn 1.524, unchanged by this fix — that
part of the pipeline was never buggy). But on the occlusion-*recovery*
comparison specifically — this project's actual thesis
("Mamba-based inertial maintenance beats GRU/Transformer under
occlusion") — **mamba and gru are statistically indistinguishable on both
synth (1.733 vs 1.732) and natural (1.786 vs 1.776) occlusion, and gru has
the smaller gap-from-clean on both axes.** mamba's overall edge looks driven
by the *clean* slice (1.259 vs 1.349), not by occlusion robustness. attn is
worst on synth as expected (smallest capacity, per caveat above) but not
worst on natural gap. Caveats on this reading: n=45 for the natural slice is
small (wide CI, don't over-read small deltas there), and caveat #1
(GT-anchor leak) and caveat #3 (recon loss diluted) still apply equally to
all three arms so the absolute numbers still carry those biases. **This is
the headline result the write-up needs to report honestly** — the
mamba-specific occlusion-robustness hypothesis is not supported by this
ablation as run; the actual finding is "recurrent state (mamba OR gru) beats
attention here," not "mamba beats gru."

## Where we are

Following the prior session's decision, went straight to the ablation
(spec 3.4.2: mamba vs gru vs attn, controlled comparison). This session's
work, in order:

1. **Fresh pod, environment was broken**: torch was 2.4.1+cu124, which does
   NOT support the RTX 5090 (sm_120 — confirmed via the exact "no kernel
   image" warning the Dockerfile predicts). Reinstalled `torch==2.8.0` from
   the cu128 index, reinstalled `blackboxprotobuf`, reinstalled the
   `mambahybrid` package (`pip install --no-deps -e .`). Verified with a GPU
   matmul + `pytest tests/` (3 passed). torchvision/torchaudio are now
   version-mismatched (still cu124) but nothing in this repo imports them —
   harmless.
2. **attn encoder OOM'd at exp5 scale (d_model 256)**: `CausalAttnEncoder`
   re-runs the whole growing sequence through every layer at every one of
   ~90 outer timesteps with the outer per-step checkpoint force-disabled for
   "attn" (`hybrid.py`, growing state breaks checkpoint replay). At d_model
   256 / depth 6 this OOMs at 32 GB even at batch_size=1. Added **per-layer**
   gradient checkpointing inside `CausalAttnEncoder.step`
   (`model/encoders.py`) — checkpoints each `TransformerEncoderLayer` call
   individually (safe: no state spans steps within one call). This alone
   still wasn't enough at d_model 256 (<10 steps/240s, i.e. days for 20k
   steps) — **decision: keep attn at its original scale (d_model 128, depth
   4, decoder_layers 2, n_heads 4, 2.28 M params vs mamba/gru's ~12-14 M)**.
   This is the ablation's known, now-measured fairness caveat (spec 3.4.2 /
   prior handoff already flagged "attn can't use grad_checkpoint/compile —
   match what you can"). Measured speed: attn 0.64 it/s (batch 2, d_model
   128) vs gru 0.9-1.0 it/s (batch 4, d_model 256, compile on) vs mamba 1.15
   it/s (exp5, batch 4, d_model 256, compile on).
3. **Step budget — settled on 20,000 for all three arms**, each trained
   fresh from scratch (not reusing exp5's checkpoint). Reasoning: train.py's
   LR schedule normalizes decay progress by `max_steps`
   (`prog = (step - warmup_iters) / (max_steps - warmup_iters)`), so
   exp5's `best.pt` (step 26000 out of a 35000-step schedule) is only 73.5%
   decayed — reusing it would compare an under-annealed mamba checkpoint
   against gru/attn checkpoints that decay fully to their own endpoint. A
   fresh 20k run for mamba costs ~4.8h but removes this confound entirely;
   worth it for a portfolio piece where a reviewer could poke at exactly
   this detail. (lr for mamba is fixed at 2e-4 — exp5's own sweep winner,
   not re-swept.)
4. **Config updates**: `configs/ablation_gru.yaml` / `configs/ablation_attn.yaml`
   now mirror exp5's loss weights / EWTA / data settings, at
   `max_steps: 20000`; attn kept at d_model 128 per point 2 above (see the
   file headers for the full rationale — written like exp5.yaml's own
   comments, meant to stay in the repo).
5. **Wrote `scripts/run_ablation.py`** — the actual driver for all of this,
   unattended and resumable:
   - gru: lr sweep {5e-5, 1e-4, 2e-4} @ 6k steps -> pick best val
     all/minADE -> full 20k run at that lr.
   - attn: same sweep protocol @ batch_size 2.
   - mamba: straight to a fresh 20k run at lr 2e-4 (no sweep — already done
     by exp5).
   - Resumable: re-running the script skips any run whose `log.jsonl`
     already reaches its target step, so a crash mid-pipeline only re-does
     the interrupted run, not everything before it.
   - At the end: runs `scripts/diagnose_ckpt.py` on each arm's `best.pt` and
     writes `runs/ablation_report.txt` with all three.

## State right now

- **`scripts/run_ablation.py` is running in the background** (`nohup`,
  started 2026-09-11 03:39, PID varies — check `ps aux | grep run_ablation`).
  Log: `runs/ablation_driver.log`. Expected total wall clock **~32h**
  (gru sweep ~5h + gru full ~6h + attn sweep ~8h + attn full ~9h + mamba full
  ~5h). It is safe to just `tail -f runs/ablation_driver.log` to check in;
  do NOT launch a second copy — check `ps aux` / the log's last line first.
- If it crashed or you need to resume after a pod restart: just rerun
  `nohup python -u scripts/run_ablation.py > runs/ablation_driver.log 2>&1 &`
  — it will skip everything already completed (checked via `log.jsonl`'s max
  logged step vs. that stage's target step) and continue from where it
  stopped.
- Sweep run dirs: `runs/ablation_{gru,attn}_sweep_lr{5e-5,1e-4,2e-4}/`. Full
  run dirs: `runs/ablation_{gru,attn,mamba}/`. Final configs actually used:
  `configs/ablation_{gru,attn,mamba}_final.yaml` (written by the driver once
  each stage's winner lr is picked — not present until that stage runs).
- exp5 (mamba, 35k steps, lr 2e-4) is still on disk at
  `runs/exp5_scale_0910/` (best.pt minADE 1.25 @ step 26000) — kept for
  reference/comparison but is NOT the mamba arm anymore (see point 3 above).

## After the driver finishes (~32h from launch)

- Read `runs/ablation_report.txt` (written automatically) — multimodality,
  CV-baseline comparison, ADE-by-horizon, occlusion slice for each arm.
- Build the actual ablation table (spec 3.4.3): minADE_6/minFDE_6/MissRate@2m
  on the shared val split, **plus the occlusion-slice numbers** (synthetic +
  natural gap minADE/FDE) for each of mamba/gru/attn — this occlusion
  comparison is the actual thesis claim and the headline result.
- Document the two fairness caveats plainly in the write-up: attn's
  much smaller capacity (2.28M vs ~12-14M params) and smaller batch (2 vs
  4) — both structurally forced by attn's O(T^2) memory, not a choice.
- Portfolio polish (still open, unchanged from before): qualitative
  trajectory visualizations, and the hypothesis -> system -> ablation ->
  honest-limitations write-up.

## How to run (unchanged basics)

```bash
cd /workspace/inertnet
# fresh pod: torch must be >=2.7 for the 5090 (sm_120) — check `python3 -c
# "import torch; print(torch.__version__)"` first; if it says cu12x < 2.7,
# reinstall: pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt && pip install --no-deps blackboxprotobuf==1.0.1 && pip install --no-deps -e .

# check on the ablation driver
tail -f runs/ablation_driver.log
ps aux | grep run_ablation

# diagnose any finished checkpoint
python scripts/diagnose_ckpt.py --ckpt runs/ablation_gru/best.pt --config configs/ablation_gru_final.yaml
```
- **Kill everything**: `pkill -9 -f "run_ablation\.py"` then
  `pkill -9 -f "mambahybrid\.train"` (note the escaped dots — `pkill -f`
  matches full command lines, and an *unescaped* pattern that appears
  verbatim in your own shell invocation will self-match and kill your own
  shell before it runs anything else. Learned this the hard way this
  session — always escape dots in `-f` patterns for this codebase's module
  path.) Then check `nvidia-smi`.
