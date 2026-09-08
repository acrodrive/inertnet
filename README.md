# MambaHybrid

Occlusion-robust multi-agent trajectory prediction on the **Waymo Open Motion
Dataset**. A per-step selective-SSM (Mamba) recurrence maintains each object's
physical state through sensor drop-outs; a Transformer decoder then fuses that
state with vectorized map + live traffic-signal context. Trajectories are
decoded one-shot, K=6 modes, from every timestep.

See [`Specification.md`](Specification.md) for the full design rationale.

## Layout

```
src/mambahybrid/
  data/
    womd_proto.py    field map + decoders for the Scenario protobuf (no TF)
    parser.py        TFRecord -> Scenario (numpy)
    map_vectorize.py polyline transform + resample
    preprocess.py    Scenario -> one training sample (anchor frame, slots,
                     synthetic occlusion, static map, per-step signals)
    dataset.py       torch Dataset + on-disk .npz cache + collate
  model/
    mamba_block.py   pure-torch selective SSM with a recurrent .step()
    encoders.py      mamba | gru | attn  (ablation, Spec 3.4.2)
    modules.py       ObjectEmbed, MapEncoder (static/dynamic), CorrectionGate,
                     MapDecoder
    heads.py         PriorHead, GTHead, TrajectoryHead
    hybrid.py        the per-step time loop (Loop 1) + one-shot future decode
  losses.py          6-term total loss (Spec 3.2.6) + masking
  metrics.py         minADE/minFDE/MissRate + occlusion slices
  train.py / eval.py entry points
```

## Data

Only the WOMD **scenario** protos are needed (no camera/lidar). Point the config
at the shards:

```
dataset/Waymo/training/training.tfrecord-XXXXX-of-01000
```

`python scripts/inspect_shard.py "<glob>"` prints occlusion / signal stats.

## Run

Local smoke test (CPU):

```
pip install -e .
PYTHONPATH=src python tests/test_pipeline.py
```

Train on RunPod / RTX 4090. Code, data and outputs all live on the network
volume (mounted at `/workspace`); the model is updated with `git pull`, never an
image rebuild.

Simplest path — a **stock RunPod PyTorch template** (SSH ready out of the box):

```
# on a fresh pod, network volume at /workspace:
cd /workspace && git clone https://github.com/acrodrive/inertnet.git
bash inertnet/scripts/pod_setup.sh     # clone/pull, pip install, link package
wandb login
#   data -> /workspace/inertnet/dataset/Waymo/training/training.tfrecord-*
python -m mambahybrid.train --config configs/default.yaml
```

`pod_setup.sh` points the pip cache at `/workspace/.pip-cache`, so the install is
only slow on the very first pod.

Faster pod cold-starts — build the deps image once (`Dockerfile` bakes deps +
`scripts/start.sh`, which starts sshd and keeps the pod alive); `pod_setup.sh`
still does the git clone + package link on the pod:

```
docker build -t <you>/inertnet:deps . && docker push <you>/inertnet:deps
```

SSH: add your **public key** in RunPod → Settings → SSH Public Keys. The web
terminal and proxy SSH (`ssh.runpod.io`) need no exposed port; **VS Code
Remote-SSH, `scp`, `rsync` and `ssh -L` tunnels need direct SSH** — set **Expose
TCP Ports: 22** on the pod (Secure Cloud, which has a public IP). The Connect
dialog then shows `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`.

Monitoring: set `wandb: true` in the config (already on in `configs/*.yaml`).
Runs log to project `inertnet`; the wandb run id is derived from `ckpt_dir`, so
`--resume runs/exp1/last.pt` reattaches to the same run after a pod restart.

Ablation (the key comparison — Mamba removed):

```
python -m mambahybrid.train --config configs/ablation_attn.yaml
```

Evaluate a checkpoint (prints `all` / `occluded` / `clean` slices):

```
python -m mambahybrid.eval --config configs/default.yaml --ckpt runs/exp1/best.pt
```

## Notes

- `mamba_block.py` is pure PyTorch and steps sequentially because the corrected
  feature `f_{t-1}` is fed back as the next input — the sequence cannot be
  scanned in parallel. On GPU you may install `mamba-ssm` and swap in its
  `.step()`; the interface matches.
- Geometry is normalised (`pos_scale`, `vel_scale`, `size_scale`); metrics
  multiply back to metres.
- `grad_checkpoint: true` checkpoints each timestep — needed for the full
  `n_slots=300`, 91-step BPTT on 24 GB.
