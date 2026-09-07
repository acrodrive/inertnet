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
python -c "import einops, wandb" 2>/dev/null || pip install -r requirements.txt
python -c "import blackboxprotobuf"  2>/dev/null || pip install --no-deps blackboxprotobuf==1.0.1
pip install --no-deps -e .          # link the mambahybrid package

echo "==> sanity check"
python - <<'PY'
import torch, blackboxprotobuf, wandb
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY

cat <<'EOF'

next:
  wandb login                        # or: export WANDB_API_KEY=...
  # data -> /workspace/inertnet/dataset/Waymo/training/training.tfrecord-*
  python -m mambahybrid.train --config configs/default.yaml
EOF
