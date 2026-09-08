# InertNet — optional dependency image for RunPod / RTX 4090 (Ada, sm_89).
#
# You usually DON'T need this: a stock RunPod "PyTorch" template already has
# CUDA + sshd, and `scripts/pod_setup.sh` installs the rest onto the network
# volume (with the pip cache on /workspace, so it's only slow on the first pod).
#
# Build this only if you want faster pod cold-starts. It bakes deps ONLY — code,
# data and outputs live on the network volume (mounted at /workspace), updated
# with `git pull`, never an image rebuild. scripts/start.sh is the entrypoint:
# it injects RunPod's $PUBLIC_KEY, starts sshd (for VS Code Remote-SSH etc.),
# and keeps the container alive.
#
#   docker build -t <you>/inertnet:deps . && docker push <you>/inertnet:deps

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build openssh-server tmux && \
    rm -rf /var/lib/apt/lists/*

# Deps only — no application code is copied in. blackboxprotobuf is installed
# --no-deps because its metadata pins protobuf==3.10.0 (it only uses the stable
# google.protobuf.internal wire codecs, which work on modern protobuf).
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
 && pip install --no-deps blackboxprotobuf==1.0.1

# Optional fused Mamba kernels (needs this -devel base for nvcc):
# RUN pip install "causal-conv1d>=1.4.0" "mamba-ssm>=2.2.2"

COPY scripts/start.sh /start.sh
RUN chmod +x /start.sh

WORKDIR /workspace
CMD ["/start.sh"]
