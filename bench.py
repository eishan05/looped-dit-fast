"""Forward-pass time per loop depth, eager vs compiled.

One forward = one of the 2 x steps model calls a CFG sample makes. Loop depths are measured
round-robin so background load hits them evenly; the median of --reps rounds is reported.

    python bench.py --checkpoint looped-dit-b32.pt --loops 1 2 3 4 6 8 --batches 1 8
"""

from __future__ import annotations

import argparse
import statistics

import torch

from fast_looped_dit import FastLoopedDiT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--loops", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--steps", type=int, default=100, help="for the per-image estimate")
    a = p.parse_args()

    pipe = FastLoopedDiT(a.checkpoint, compile=True)
    model, cfg = pipe.model, pipe.cfg
    print("mode     batch  loops  blocks  ms/forward  s/image")
    for b in a.batches:
        ids, mask = pipe.text_encoder.tokenize(["a red cube on top of a blue sphere"] * b)
        text = pipe.text_encoder.encode(ids, mask).to(torch.bfloat16)
        x = torch.randn(b, 3, cfg.image_size, cfg.image_size, device=pipe.device, dtype=torch.bfloat16)
        for n in a.loops:
            pipe._prepare(b, n)
        for mode, fn in (("eager", model), ("compiled", pipe.compiled)):
            def call(n):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    return fn(x, text, mask, num_loops=n)

            for n in a.loops:
                for _ in range(3):
                    call(n)
            torch.cuda.synchronize()
            times = {n: [] for n in a.loops}
            for _ in range(a.reps):
                for n in a.loops:
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    s.record()
                    for _ in range(5):
                        call(n)
                    e.record()
                    torch.cuda.synchronize()
                    times[n].append(s.elapsed_time(e) / 5)
            for n in a.loops:
                ms = statistics.median(times[n])
                blocks = model.pre + model.core * n + model.post
                print(f"{mode:8} {b:5}  {n:5}  {blocks:6}  {ms:10.2f}  {2 * a.steps * ms / 1000 / b:7.3f}", flush=True)


if __name__ == "__main__":
    main()
