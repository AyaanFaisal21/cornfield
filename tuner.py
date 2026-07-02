"""Autotuning harness: generate kernel variants, compile, correctness-check,
benchmark, and pick the winner for the given shape + GPU.

Thesis (learned the hard way in the TransformerOp project): the fastest kernel
depends on shape AND hardware, and you can't reliably predict it -- so generate
candidate kernels, measure them on the real device, and select. That's exactly
why cuBLAS / cuDNN / Triton autotune.

Entry points:
  - tune()          : the front door -- cache lookup, then auto-picks grid vs
                      random by whether the space fits the budget; caches the winner
                      keyed on op | bucketed-shape | gpu | template-hash.
  - autotune()      : bare grid search -- try every config in a list.
  - random_search() : bare random search -- sample a budget from a large space.
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
    Returns (label, seconds, ok). Incorrect or un-launchable kernels get time = inf
    and ok = False so the search skips them instead of crashing -- a real search
    space includes configs that fail to compile or exceed hardware limits (registers,
    threads, shared memory)."""
    label = "_".join(f"{k}{v}" for k, v in cfg.items())
    src = template
    for k, v in cfg.items():
        src = src.replace(f"__{k}__", str(v))
    path = GEN / f"{name}_{label}.cu"
    path.write_text(src)
    try:
        ext = load(name=f"{name}_{label}", sources=[str(path)], verbose=False)
        fn = getattr(ext, run)
        out = fn(*inputs)
        if DEVICE == "cuda":
            torch.cuda.synchronize()        # surface async launch errors here, in the try
        ok = torch.allclose(out, ref, rtol=rtol, atol=atol)
        t = benchmark(lambda: fn(*inputs)) if ok else float("inf")
        return label, t, ok
    except Exception as e:
        print(f"  {label:24s}  -- skipped ({str(e).splitlines()[0][:50]})")
        return label, float("inf"), False


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


def bucket_shape(shape):
    """Round each dim up to the next power of two. Nearby shapes (1000 vs 1024)
    almost always share a winning config, so exact-shape cache keys would force a
    full re-search for every slightly-new shape. Bucketing trades a little
    precision at bucket boundaries for far more cache hits (force=True re-tunes)."""
    return tuple(1 << max(0, (int(d) - 1).bit_length()) for d in shape)


def cache_key(name, shape, template):
    """Key on op + shape + GPU + a template hash. The template hash auto-invalidates
    the cache when you change the kernel, so you never get a stale config."""
    th = hashlib.md5(template.encode()).hexdigest()[:8]
    return f"{name}|{'x'.join(map(str, shape))}|{_gpu()}|{th}"


def tune(name, template, space, inputs, ref, shape=None, budget=8, force=False,
         seed=0, run="run", rtol=1e-3, atol=1e-2):
    """The front door: cache lookup -> auto strategy -> search -> cache store.

    Strategy picks itself: grid and random search are the same loop -- evaluate a
    set of configs, keep the fastest -- differing only in whether that set is the
    whole space or a sample. So if the space fits in the budget, run it all
    (exhaustive, proven-best-in-space); otherwise sample `budget` configs.

    The cache key uses the BUCKETED shape (see bucket_shape), so a shape near one
    already tuned reuses its config instantly. Returns the best config dict."""
    if shape is None:   # default: every dim of every tensor input identifies the problem size
        shape = tuple(d for t in inputs if hasattr(t, "shape") for d in t.shape)

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    key = cache_key(name, bucket_shape(shape), template)
    if not force and key in cache:
        e = cache[key]
        print(f"  cache HIT  [{key}]\n    -> {e['config']}  ({e['time_ms']:.3f} ms, no search)")
        return e["config"]

    if len(space) <= budget:
        picks, strategy = list(space), f"grid ({len(space)} configs)"
    else:
        picks = random.Random(seed).sample(space, budget)
        strategy = f"random {budget} of {len(space)}"
    print(f"  cache MISS [{key}] -- strategy: {strategy}")

    GEN.mkdir(exist_ok=True)
    best = None  # (cfg, label, seconds)
    for i, cfg in enumerate(picks, 1):
        label, t, ok = _eval_config(name, template, cfg, inputs, ref, run, rtol, atol)
        if ok and (best is None or t < best[2]):
            best = (cfg, label, t)
        bs = f"{best[2]*1e3:7.3f}" if best else "    inf"
        print(f"  [{i:2d}/{len(picks)}] {label:24s} {t*1e3:8.3f} ms   best-so-far {bs} ms")
    if best is None:
        raise RuntimeError(f"tune({name}): no config compiled, launched, and passed allclose")

    cfg, label, secs = best
    cache[key] = {"config": cfg, "label": label, "time_ms": secs * 1e3,
                  "strategy": strategy, "shape": list(shape)}
    CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))
    print(f"\n  best: {label}  ({secs*1e3:.3f} ms)  -- cached for [{key}]")
    return cfg
