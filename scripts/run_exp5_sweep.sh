#!/usr/bin/env bash
# exp5 lr sweep: 3 short runs back-to-back on the one GPU. ~100 min each.
set -uo pipefail
cd /workspace/inertnet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for cfg in configs/exp5_lr5e-5.yaml configs/exp5_lr1e-4.yaml configs/exp5_lr2e-4.yaml; do
    name=$(basename "$cfg" .yaml)
    dir="runs/${name}"
    mkdir -p "$dir"
    echo "=== $(date -u +%H:%M:%S)  starting $name ==="
    python -m mambahybrid.train --config "$cfg" > "$dir/train.log" 2>&1
    echo "=== $(date -u +%H:%M:%S)  $name exit $? ==="
done
echo "=== sweep done ==="
