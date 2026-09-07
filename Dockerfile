# RunPod / RTX 4090 (Ada, sm_89). CUDA 12.4 + PyTorch 2.4+.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
WORKDIR /workspace/mambahybrid

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional fused kernels (safe to remove if the build is slow / unneeeded):
# RUN pip install --no-cache-dir causal-conv1d>=1.4.0 mamba-ssm>=2.2.2

COPY . .
ENV PYTHONPATH=/workspace/mambahybrid/src

# Data is mounted at runtime, e.g. -v /runpod-volume/waymo:/workspace/mambahybrid/dataset/Waymo
CMD ["python", "-m", "mambahybrid.train", "--config", "configs/default.yaml"]
