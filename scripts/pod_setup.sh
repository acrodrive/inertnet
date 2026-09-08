#!/usr/bin/env bash
# One-time setup on a fresh RunPod pod. Works on either:
#   - a stock RunPod PyTorch template (installs deps here), or
#   - the custom deps image from this repo's Dockerfile (deps already baked).
#
# Assumes the network volume is mounted at /workspace (persistent, portable).
# The container disk (everything outside /workspace) is ephemeral, so the pip
# cache is pointed at the volume to make re-installs on later pods fast.
#
#   bash scripts/pod_setup.sh
#
# Re-running is safe: git pull + re-link + skip already-satisfied installs.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/acrodrive/inertnet.git}"
WORK="${WORK:-/workspace}"
DIR="$WORK/inertnet"

export PIP_CACHE_DIR="$WORK/.pip-cache"

# Find a Python. PyTorch images keep it in /opt/conda; some stock templates use
# a venv that a non-login shell hasn't activated. Never rely on bare `pip`.
if ! command -v python >/dev/null 2>&1; then
    for d in /opt/conda/bin /venv/bin /workspace/venv/bin; do
        [ -x "$d/python" ] && export PATH="$d:$PATH" && break
    done
fi
PY="$(command -v python || command -v python3 || true)"
[ -n "$PY" ] || { echo "no python on PATH — run: which -a python3; ls /opt/conda/bin" >&2; exit 1; }
PIP="$PY -m pip"
echo "python: $PY"

cd "$WORK"
if [ -d "$DIR/.git" ]; then
    echo "==> git pull"
    git -C "$DIR" pull --ff-only
else
    echo "==> git clone $REPO_URL"
    git clone "$REPO_URL" "$DIR"
fi
cd "$DIR"

echo "==> deps"
"$PY" -c "import einops, wandb" 2>/dev/null || $PIP install -r requirements.txt
"$PY" -c "import blackboxprotobuf" 2>/dev/null || $PIP install --no-deps blackboxprotobuf==1.0.1
$PIP install --no-deps -e .          # link the mambahybrid package

echo "==> sanity check"
"$PY" - <<'PY'
import torch, blackboxprotobuf, wandb
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY

# Put the interpreter's bin dir on PATH for future shells in this pod (the
# pip/wandb/python console scripts live there; ~/.bashrc is ephemeral, so this
# re-runs per pod).
BINDIR="$(cd "$(dirname "$PY")" && pwd)"
grep -qF "$BINDIR" ~/.bashrc 2>/dev/null || echo "export PATH=\"$BINDIR:\$PATH\"" >> ~/.bashrc

cat <<EOF

done. for THIS shell run:  export PATH="$BINDIR:\$PATH"
     (new shells pick it up from ~/.bashrc automatically)

next:
  export WANDB_API_KEY=...            # from wandb.ai/authorize  (or: wandb login)
  # data -> /workspace/inertnet/dataset/Waymo/training/training.tfrecord-*
  tmux new -s train
  python -m mambahybrid.train --config configs/default.yaml
EOF
