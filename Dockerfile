# InertNet — dependency image for RunPod / RTX 4090 (Ada, sm_89).
#
# This image bakes the Python deps ONLY. Code + data + outputs live on the
# RunPod network volume (mounted at /workspace), so the image never needs a
# rebuild when the model changes — you `git pull` on the pod instead.
#
#   Build & push once:
#     docker build -t <you>/inertnet:deps .
#     docker push <you>/inertnet:deps
#
#   On a fresh pod (network volume at /workspace), bootstrap once:
#     cd /workspace && git clone https://github.com/acrodrive/inertnet.git \
#       && bash inertnet/scripts/pod_setup.sh
#   After that, `bash scripts/pod_setup.sh` just pulls + relinks.

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Deps only — no application code is copied in.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Optional fused Mamba kernels (needs this -devel base for nvcc). Uncomment to
# bake them in; adds a few minutes to the build.
# RUN pip install "causal-conv1d>=1.4.0" "mamba-ssm>=2.2.2"

WORKDIR /workspace
CMD ["bash"]
