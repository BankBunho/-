# The validator builds this file for every image task (standalone-image-toolkit-trainer.dockerfile: all non-flux types;
# standalone-image-trainer.dockerfile: flux) - both are this same lean image; the flux one also bakes the evaluator's
# own flux text encoders (below).
# Exactly the validator's evaluation stack (G.O.D validator-diffusion.dockerfile: python 3.11, ComfyUI 694815f4,
# torch 2.9.1 cu128, diffusers 0.39.0, transformers 5.10.2, huggingface-hub 1.18.0, safetensors 0.8.0, pydantic 2.13.4,
# accelerate 1.6.0) + our ComfyUI-native trainer. No ai-toolkit: cold builds of heavy hybrid images reached the
# validator's 1800 s build limit (another miner, 21 Sep). uv instead of pip for fast parallel installs.
# INTEGRITY DECLARATION: only the evaluator's own public VAE files and (this flux image) the evaluator's own public flux
# text encoders are baked (sha256-pinned); base model and data come solely from the validator-staged /cache.
FROM python:3.11-slim
ARG COMFYUI_COMMIT=694815f498295080a0e15a1502edc9dba841b110
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1 PIP_NO_CACHE_DIR=1 UV_NO_CACHE=1 \
    COMFY_ROOT=/opt/ComfyUI G56CT_BAKED=/opt/g56/baked G56CT_CKPT_ROOT=/app/checkpoints UV_HTTP_TIMEOUT=120
WORKDIR /app
# The validator builds this image cold for every task and fails the task on any build error (no retry), so every
# network step retries: uv's 30 s default read timeout cut a cold build on 2 Oct (nvidia-cuda-nvrtc wheel).
RUN pip install --no-cache-dir uv==0.8.22 && \
    for i in 1 2 3; do \
        uv pip install --system --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 && break; \
        [ $i = 3 ] && exit 1; echo "torch install attempt $i failed, retrying"; sleep 20; \
    done
RUN for i in 1 2 3 4; do \
    python -c "import io, tarfile, urllib.request, os; \
u = 'https://codeload.github.com/comfyanonymous/ComfyUI/tar.gz/${COMFYUI_COMMIT}'; \
b = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'g56-build'}), timeout=300).read(); \
t = tarfile.open(fileobj=io.BytesIO(b)); root = t.getmembers()[0].name.split('/')[0]; t.extractall('/opt'); \
os.rename('/opt/' + root, '/opt/ComfyUI')" && break; \
    [ $i = 4 ] && exit 1; echo "ComfyUI download attempt $i failed, retrying"; rm -rf /opt/ComfyUI /opt/ComfyUI-*; sleep 15; \
    done && \
    for i in 1 2 3; do \
        uv pip install --system -r /opt/ComfyUI/requirements.txt \
            diffusers==0.39.0 transformers==5.10.2 huggingface-hub==1.18.0 safetensors==0.8.0 pydantic==2.13.4 accelerate==1.6.0 && break; \
        [ $i = 3 ] && exit 1; echo "requirements install attempt $i failed, retrying"; sleep 20; \
    done && \
    python -c "import torch, diffusers, transformers, comfy_kitchen; assert torch.__version__.startswith('2.9.1'), torch.__version__"
COPY ops/docker/bake_comfy_assets.py /tmp/bake_comfy_assets.py
RUN python /tmp/bake_comfy_assets.py /opt/g56/baked --flux-te /app/flux && mkdir -p /app/checkpoints /opt/comfy-venv/bin && \
    ln -s /usr/local/bin/python /opt/comfy-venv/bin/python
# flux: the evaluator's own fp16 TEs (comfyanonymous/flux_text_encoders clip_l + t5xxl_fp16 at a pinned revision) sit in
# /app/flux = FAMILIES['flux']['te_files']: the conditioning is the evaluator's DualCLIPLoader output bit for bit, and a
# one-file flux repo never borrows another cached model's text_encoder folder (audit T008 P0).
COPY comfytrain /workspace/comfytrain
ENTRYPOINT ["python", "/workspace/comfytrain/g56_comfy_train.py"]
