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

import hashlib
import itertools
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEN = Path(__file__).parent / ".gen"        # generated .cu variants land here
CACHE = Path(__file__).parent / "cache.json"  # tuned configs, keyed by op|shape|gpu|template


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
    best = None  # (cfg, label, seconds)
    for i, cfg in enumerate(picks, 1):
        label, t, ok = _eval_config(name, template, cfg, inputs, ref, run, rtol, atol)
        if ok and (best is None or t < best[2]):
            best = (cfg, label, t)
        bs = f"{best[2]*1e3:7.3f}" if best else "    inf"
        print(f"  [{i:2d}/{len(picks)}] {label:24s} {t*1e3:8.3f} ms   best-so-far {bs} ms")
    if best:
        print(f"\n  best: {best[1]}  ({best[2]*1e3:.3f} ms)  "
              f"-- {len(picks)} trials over a {len(space)}-config space")
    return best  # (cfg dict, label, seconds)


# ---- config cache: tune once per (op, shape, gpu, template), then look it up ----

def _gpu():
    return torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"


def cache_key(name, shape, template):
    """Key on op + shape + GPU + a template hash. The template hash auto-invalidates
    the cache when you change the kernel, so you never get a stale config."""
    th = hashlib.md5(template.encode()).hexdigest()[:8]
    return f"{name}|{'x'.join(map(str, shape))}|{_gpu()}|{th}"


def autotune_cached(name, template, space, inputs, ref, shape, budget=8,
                    force=False, seed=0, run="run", rtol=1e-3, atol=1e-2):
    """Look up the best config for (op, shape, gpu, template); search + store on a miss."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    key = cache_key(name, shape, template)

    if not force and key in cache:
        e = cache[key]
        print(f"  cache HIT  [{key}]\n    -> {e['config']}  ({e['time_ms']:.3f} ms, no search)")
        return e["config"]

    print(f"  cache MISS [{key}] -- searching")
    best = random_search(name, template, space, inputs, ref, budget,
                         seed=seed, run=run, rtol=rtol, atol=atol)
    cfg, label, secs = best
    cache[key] = {"config": cfg, "label": label, "time_ms": secs * 1e3}
    CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))
    print(f"  cached best for [{key}]")
    return cfg
