#!/usr/bin/env bash
# One-time setup on a fresh RunPod pod.
#
# Assumes: the network volume is mounted at /workspace (persistent, portable),
# and the pod runs the InertNet deps image (torch + wandb + blackboxprotobuf
# already installed). The container disk (everything outside /workspace) is
# ephemeral — nothing here is written to it.
#
#   bash scripts/pod_setup.sh
#
# Re-running is safe: it just `git pull`s and re-links the package.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/acrodrive/inertnet.git}"
WORK="${WORK:-/workspace}"
DIR="$WORK/inertnet"

cd "$WORK"
if [ -d "$DIR/.git" ]; then
    echo "==> git pull"
    git -C "$DIR" pull --ff-only
else
    echo "==> git clone $REPO_URL"
    git clone "$REPO_URL" "$DIR"
fi
cd "$DIR"

# Package is linked in editable mode; deps are already in the deps image, so
# skip them here. Fall back to installing them if this is a bare base image.
echo "==> pip install -e . (no deps)"
pip install --no-deps -e .
python -c "import wandb, einops" 2>/dev/null || pip install -r requirements.txt
python -c "import blackboxprotobuf" 2>/dev/null || pip install --no-deps blackboxprotobuf==1.0.1

echo "==> sanity check"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY

cat <<'EOF'

next:
  wandb login                        # or export WANDB_API_KEY=...
  # data -> /workspace/inertnet/dataset/Waymo/training/training.tfrecord-*
  python -m mambahybrid.train --config configs/default.yaml
EOF
