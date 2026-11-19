"""Compiled inference for Looped-DiT (https://github.com/OpenSenseNova/Looped-DiT).

The official sampler runs the model eagerly. At batch 1 the model is small enough that most
of the time goes to launching GPU kernels from Python (an RTX 5090 sits at ~34% utilisation).
Compiling with CUDA graphs (`torch.compile(mode="reduce-overhead")`) removes that overhead:
~10x faster at batch 1, with images that match eager to within bf16 rounding.

Three things are needed for that to work with the upstream code:

1. `looped_dit.model` caches its RoPE tables in a module-global dict. If that cache is first
   filled inside a CUDA graph, the cached tensors are graph outputs that the next replay
   overwrites ("accessing tensor output of CUDAGraphs that has been overwritten"). We run one
   eager forward per shape first so the cache holds ordinary tensors.
2. Every (loop count, batch size) pair is its own graph. The default Dynamo recompile limit
   (8) is too low for a loop-depth sweep, so we raise it.
3. The sampler calls the model twice per step (conditional, then unconditional). With CUDA
   graphs the second replay reuses the first's output buffer, so each output is cloned.

Usage:

    from fast_looped_dit import FastLoopedDiT
    pipe = FastLoopedDiT("looped-dit-b32.pt")
    images = pipe.generate(["a red cube on top of a blue sphere"], num_loops=4, seed=0)
"""

from __future__ import annotations

import torch
from torch import nn

from looped_dit.diffusion import euler_sample
from looped_dit.pipeline import TextEncoder, load_model, to_pil


class _Graphed(nn.Module):
    """Compiled wrapper with the same call signature as LoopedMMDiT."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.inner = model
        self.compiled = torch.compile(model, mode="reduce-overhead", fullgraph=True, dynamic=False)

    def forward(self, *args, **kwargs):
        torch.compiler.cudagraph_mark_step_begin()
        return self.compiled(*args, **kwargs).clone()  # fix 3


class FastLoopedDiT:
    def __init__(self, checkpoint: str, device: str = "cuda", compile: bool = True, text_encoder: str | None = None):
        self.device = torch.device(device)
        torch.backends.cuda.matmul.allow_tf32 = True  # as in upstream training and evaluation
        torch.backends.cudnn.allow_tf32 = True
        torch._dynamo.config.recompile_limit = 64  # fix 2
        self.model, self.cfg = load_model(checkpoint, self.device)  # EMA weights, bf16
        self.text_encoder = TextEncoder(text_encoder or self.cfg.text_encoder, self.cfg.prompt_length, self.device)
        self.compiled = _Graphed(self.model) if compile else None
        self._warm: set[tuple[int, int]] = set()

    def _prepare(self, batch: int, num_loops: int) -> None:
        """Fix 1: one eager forward at this shape fills the global RoPE cache outside any graph."""
        if (batch, num_loops) in self._warm:
            return
        size = self.cfg.image_size
        x = torch.zeros(batch, 3, size, size, device=self.device, dtype=torch.bfloat16)
        text = torch.zeros(batch, self.cfg.prompt_length, self.model.txt_embed.in_features, device=self.device, dtype=torch.bfloat16)
        mask = torch.ones(batch, self.cfg.prompt_length, device=self.device, dtype=torch.long)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.model(x, text, mask, num_loops=num_loops)
        self._warm.add((batch, num_loops))

    @torch.no_grad()
    def generate(self, prompts: list[str], num_loops: int = 4, seed: int | None = None, steps: int = 100, cfg_scale: float = 6.0):
        """One PIL image per prompt. The first call at a new (batch, loops) shape compiles
        and captures the graph (~40-120 s); later calls replay it."""
        model = self.model
        if self.compiled is not None:
            self._prepare(len(prompts), num_loops)
            model = self.compiled
        if seed is not None:
            torch.manual_seed(seed)
        ids, mask = self.text_encoder.tokenize(prompts)
        text = self.text_encoder.encode(ids, mask).to(torch.bfloat16)
        x = euler_sample(model, text, mask, self.cfg.image_size, steps, cfg_scale, self.cfg.noise_scale, num_loops)
        return to_pil(x)
