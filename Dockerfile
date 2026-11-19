# Looped-DiT with a compiled (CUDA graph) sampler. Tested on an RTX 5090 (sm_120), CUDA 12.8.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Official Looped-DiT code, pinned. transformers must stay <5 (the upstream README says 5.x
# tokenizes trailing whitespace differently from the trained models).
ARG LOOPED_DIT_COMMIT=18ad35c0e35857fd3de8ba2fa8be8e0ac27eb341
RUN git clone https://github.com/OpenSenseNova/Looped-DiT /opt/Looped-DiT \
    && git -C /opt/Looped-DiT checkout "$LOOPED_DIT_COMMIT"

RUN pip install --no-cache-dir \
      transformers==4.57.1 \
      sentencepiece==0.2.0 \
      "huggingface_hub[cli]==0.36.0"

ENV PYTHONPATH=/opt/Looped-DiT:/app
COPY fast_looped_dit.py sample.py bench.py /app/
WORKDIR /work
