"""Generate images at one or more loop depths with the compiled sampler.

    python sample.py --checkpoint looped-dit-b32.pt --prompt "three yellow apples on a wooden table" \
        --loops 1 2 3 4 --out apples.png

One row per loop depth, one column per prompt. Every depth starts from the same noise.
"""

from __future__ import annotations

import argparse
import time

import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image

from fast_looped_dit import FastLoopedDiT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True, action="append", help="repeat for several prompts")
    p.add_argument("--loops", type=int, nargs="+", default=[4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg-scale", type=float, default=6.0)
    p.add_argument("--eager", action="store_true", help="skip torch.compile (the official behaviour)")
    p.add_argument("--out", default="samples.png")
    a = p.parse_args()

    pipe = FastLoopedDiT(a.checkpoint, compile=not a.eager)
    images = []
    for n in a.loops:
        pipe.generate(a.prompt, num_loops=n, seed=a.seed, steps=2, cfg_scale=a.cfg_scale)  # compile/capture
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        images += pipe.generate(a.prompt, num_loops=n, seed=a.seed, steps=a.steps, cfg_scale=a.cfg_scale)
        torch.cuda.synchronize()
        print(f"loops={n}: {time.perf_counter() - t0:.2f} s for {len(a.prompt)} image(s)", flush=True)
    save_image(torch.stack([pil_to_tensor(im) for im in images]).float() / 255, a.out, nrow=len(a.prompt))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
