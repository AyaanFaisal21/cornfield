"""Autotuning harness: generate kernel variants, compile, correctness-check,
benchmark, and pick the winner for the given shape + GPU.

Thesis (learned the hard way in the TransformerOp project): the fastest kernel
depends on shape AND hardware, and you can't reliably predict it -- so generate
candidate kernels, measure them on the real device, and select. That's exactly
why cuBLAS / cuDNN / Triton autotune.

Search strategies:
  - autotune()      : grid -- try every config in a list (fine for small spaces).
  - random_search() : sample a budget of configs from a large space (scales).
"""

import itertools
import random
import statistics
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEN = Path(__file__).parent / ".gen"   # generated .cu variants land here


def benchmark(fn, warmup=10, iters=50):
    """Median wall-clock seconds per call, GPU-synchronized."""
    for _ in range(warmup):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _eval_config(name, template, cfg, inputs, ref, run, rtol, atol):
    """Substitute one config into the template, compile, check, benchmark.
    Returns (label, seconds, ok). Incorrect kernels get time = inf (never chosen)."""
    label = "_".join(f"{k}{v}" for k, v in cfg.items())
    src = template
    for k, v in cfg.items():
        src = src.replace(f"__{k}__", str(v))
    path = GEN / f"{name}_{label}.cu"
    path.write_text(src)
    ext = load(name=f"{name}_{label}", sources=[str(path)], verbose=False)
    fn = getattr(ext, run)
    out = fn(*inputs)
    ok = torch.allclose(out, ref, rtol=rtol, atol=atol)
    t = benchmark(lambda: fn(*inputs)) if ok else float("inf")
    return label, t, ok


def make_space(ranges, valid=None):
    """Cartesian product of knob value-lists -> list of config dicts, optionally
    filtered by a validity predicate (prune kernels that can't launch)."""
    keys = list(ranges)
    space = [dict(zip(keys, vals)) for vals in itertools.product(*ranges.values())]
    return [c for c in space if valid(c)] if valid else space


def autotune(name, template, configs, inputs, ref, run="run", rtol=1e-3, atol=1e-2):
    """Grid search: evaluate every config; return results fastest-first."""
    GEN.mkdir(exist_ok=True)
    results = []
    for cfg in configs:
        label, t, ok = _eval_config(name, template, cfg, inputs, ref, run, rtol, atol)
        print(f"  {label:24s}  {t*1e3:8.3f} ms   correct={ok}")
        results.append((label, t, ok))
    results.sort(key=lambda r: r[1])
    if results and results[0][2]:
        print(f"\n  best: {results[0][0]}  ({results[0][1]*1e3:.3f} ms)")
    return results


def random_search(name, template, space, inputs, ref, budget, seed=0,
                  run="run", rtol=1e-3, atol=1e-2):
    """Sample `budget` configs from `space`, evaluate them, track the running best.
    Finds a near-optimal config without compiling the whole space."""
    GEN.mkdir(exist_ok=True)
    picks = random.Random(seed).sample(space, min(budget, len(space)))
    print(f"  random search: {len(picks)} of {len(space)} configs (seed={seed})")
    best = None
    for i, cfg in enumerate(picks, 1):
        label, t, ok = _eval_config(name, template, cfg, inputs, ref, run, rtol, atol)
        if ok and (best is None or t < best[1]):
            best = (label, t, ok)
        bs = f"{best[1]*1e3:7.3f}" if best else "    inf"
        print(f"  [{i:2d}/{len(picks)}] {label:24s} {t*1e3:8.3f} ms   best-so-far {bs} ms")
    if best:
        print(f"\n  best: {best[0]}  ({best[1]*1e3:.3f} ms)  "
              f"-- {len(picks)} trials over a {len(space)}-config space")
    return best
