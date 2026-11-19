# looped-dit-fast

Compiled inference for [Looped-DiT](https://github.com/OpenSenseNova/Looped-DiT), with a loop-depth sampler and benchmark.
About 10x faster than the official sampler at one image at a time, with the same images.

Write-up: [What extra loops buy a looped image model](https://eishanlawrence.com/blog/looped-image-model).

## Why

The official sampler runs the model eagerly. The model is small, so at one image at a time most of the time goes to
launching GPU kernels from Python. An RTX 5090 sat at about 34% utilisation. `torch.compile` with CUDA graphs removes
that overhead.

Seconds per 512×512 image, Looped-DiT B/32, 100 steps, guidance 6.0, RTX 5090:

| Loops | Official (eager), 1 image | Compiled, 1 image | Compiled, batch of 8 (per image) |
|---|---|---|---|
| 1 | 5.0 | 0.56 | 0.27 |
| 2 | 6.6 | 0.68 | 0.34 |
| 3 | 8.0 | 0.82 | 0.40 |
| 4 (trained) | 9.6 | 0.94 | 0.47 |
| 6 | 12.6 | 1.19 | 0.62 |
| 8 | 15.6 | 1.47 | 0.76 |

Compiled and eager images differ by about 0.005 mean absolute pixel difference (bf16 rounding). Peak VRAM is about
2 GiB at every loop count.

## What it fixes

`fast_looped_dit.py` wraps the official pipeline. Three things were needed to compile it:

1. **RoPE cache.** `looped_dit.model` caches its rotary tables in a module-global dict. If the cache is first filled
   inside a CUDA graph, the next replay overwrites the cached tensors:
   `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten`. One eager forward per shape before
   compiling fills it with ordinary tensors.
2. **Recompile limit.** Every (loop count, batch size) pair is its own graph. Dynamo's default limit of 8 recompiles is
   too low for a loop sweep (`FailOnRecompileLimitHit` with `fullgraph=True`), so it's raised to 64.
3. **Output reuse.** Guidance calls the model twice per step. With CUDA graphs the second replay reuses the first's
   output buffer, so each output is cloned.

The first call at a new (batch, loops) shape compiles and captures, which takes about 40-120 s. Later calls replay it.

## Run

Docker, with an NVIDIA GPU:

```bash
docker build -t looped-dit-fast .
hf download sensenova/Looped-DiT-B32 looped-dit-b32.pt --local-dir checkpoints

docker run --rm --gpus all --ipc=host -v "$PWD/checkpoints":/ckpt:ro -v "$PWD/out":/work \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface looped-dit-fast \
  python /app/sample.py --checkpoint /ckpt/looped-dit-b32.pt \
    --prompt "three yellow apples on a wooden table" --loops 1 2 3 4 --out /work/apples.png

docker run --rm --gpus all --ipc=host -v "$PWD/checkpoints":/ckpt:ro \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface looped-dit-fast \
  python /app/bench.py --checkpoint /ckpt/looped-dit-b32.pt
```

FLAN-T5-Large (the text encoder) downloads on first run. `sample.py --eager` runs the official uncompiled path for
comparison.

From Python:

```python
from fast_looped_dit import FastLoopedDiT

pipe = FastLoopedDiT("checkpoints/looped-dit-b32.pt")
images = pipe.generate(["a red cube on top of a blue sphere"], num_loops=4, seed=0)
images[0].save("cube.png")
```

## Tested with

- RTX 5090 (sm_120), driver 610, WSL2 + Docker Desktop.
- `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`, transformers 4.57.1 (upstream requires <5).
- OpenSenseNova/Looped-DiT at `18ad35c`, `sensenova/Looped-DiT-B32` at `83a623e`.

Not tested on other GPUs or on B/16. B/16 uses the same code path, so it should work.

## Licence

MIT. Looped-DiT and its weights are MIT-licensed by their authors.
