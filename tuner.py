"""the engine. every other file in this repo is just a kernel template plus a config
space fed into this.

premise, learned the hard way in prev project TransformerOp: the fastest kernel depends on shape
and hardware and you can't predict it, so don't. generate variants, compile them,
make sure they're actually correct, time them on the real card, keep the winner.
same reason cuBLAS / cuDNN / triton all autotune.

tune() is the entry point you actually want (cache -> pick a strategy -> search ->
store the winner). autotune() and random_search() are the bare strategies underneath,
still used directly by the op scripts and kept as the serial baseline that
parallel_bench.py races against.
"""

import hashlib
import itertools
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEN = Path(__file__).parent / ".gen"          # generated .cu variants land here
CACHE = Path(__file__).parent / "cache.json"  # tuned winners, machine-specific so not committed


def benchmark(fn, warmup=10, iters=50):
    """median seconds per call. warmup first, sync around every timed call so we're
    timing the gpu and not just the launch queue."""
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


def _compile_config(name, template, cfg, run):
    """cpu half of one trial: fill the config into the template, write the .cu,
    compile it. no gpu work happens here, and nvcc/cl run as subprocesses (GIL
    released), so this half is safe to fan out across threads.
    returns (label, fn or None, err or None)."""
    label = "_".join(f"{k}{v}" for k, v in cfg.items())
    src = template
    for k, v in cfg.items():
        src = src.replace(f"__{k}__", str(v))   # plain str.replace, C code is full of {}
    path = GEN / f"{name}_{label}.cu"
    path.write_text(src)
    try:
        ext = load(name=f"{name}_{label}", sources=[str(path)], verbose=False)
        return label, getattr(ext, run), None
    except Exception as e:
        return label, None, str(e).splitlines()[0][:50]


def _measure(fn, inputs, ref, rtol, atol):
    """gpu half of one trial: correctness check, then the stopwatch. runs alone on
    the gpu on purpose, two kernels timed at once contaminate each other's numbers.
    returns (seconds, ok, err or None)."""
    try:
        out = fn(*inputs)
        if DEVICE == "cuda":
            torch.cuda.synchronize()    # async launch errors surface here, inside the try
        ok = torch.allclose(out, ref, rtol=rtol, atol=atol)
        t = benchmark(lambda: fn(*inputs)) if ok else float("inf")
        return t, ok, None
    except Exception as e:
        return float("inf"), False, str(e).splitlines()[0][:50]


def _eval_config(name, template, cfg, inputs, ref, run, rtol, atol):
    """one serial trial, compile then measure back to back (the original loop).
    anything that fails to compile, launch, or match the reference scores inf
    instead of crashing the search, since real spaces are full of configs that
    blow past some hardware limit (registers, threads, shared memory)."""
    label, fn, err = _compile_config(name, template, cfg, run)
    if err is None:
        t, ok, err = _measure(fn, inputs, ref, rtol, atol)
    else:
        t, ok = float("inf"), False
    if err is not None:
        print(f"  {label:24s}  -- skipped ({err})")
    return label, t, ok


def make_space(ranges, valid=None):
    """cartesian product of the knob ranges into config dicts, minus anything the
    validity check says can't launch (no point compiling those)."""
    keys = list(ranges)
    space = [dict(zip(keys, vals)) for vals in itertools.product(*ranges.values())]
    return [c for c in space if valid(c)] if valid else space


def autotune(name, template, configs, inputs, ref, run="run", rtol=1e-3, atol=1e-2):
    """plain grid search: run every config in the list, return results fastest first."""
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
    """sample `budget` configs from a space too big to grid, keep the running best.
    good configs turned out to be common in these spaces (first run found its winner
    on trial 1), so a small sample gets near-best at a fraction of the compiles."""
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


# ---- config cache: tune once per (op, shape, gpu, template), look it up after ----

def _gpu():
    return torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"


def bucket_shape(shape):
    """round every dim up to the next power of two. nearby shapes (1000 vs 1024)
    almost always want the same config, so exact-shape keys would pay a full search
    for every slightly new shape. costs some precision right at bucket edges,
    force=True retunes exactly if that ever matters."""
    return tuple(1 << max(0, (int(d) - 1).bit_length()) for d in shape)


def cache_key(name, shape, template):
    """op + shape + gpu + a hash of the template. the hash part means editing a
    kernel quietly invalidates its old entries, so stale configs never come back."""
    th = hashlib.md5(template.encode()).hexdigest()[:8]
    return f"{name}|{'x'.join(map(str, shape))}|{_gpu()}|{th}"


def tune(name, template, space, inputs, ref, shape=None, budget=8, force=False,
         seed=0, run="run", rtol=1e-3, atol=1e-2, workers=None):
    """the entry point: cache lookup, search on a miss, store the winner.

    strategy picks itself with one comparison, because grid and random are the same
    loop and only differ in whether the pick set is the whole space or a sample:
    space fits in the budget -> run all of it, otherwise sample `budget` configs.

    workers=None means one per cpu core: compile every pick in parallel first (that
    is nearly all of the wall time and it's cpu work), then benchmark one at a time
    since gpu timings must never overlap. workers=1 is the old interleaved serial
    loop, kept on purpose as the baseline (parallel_bench.py races the two).

    cache keys use the bucketed shape (see bucket_shape). returns the winning config."""
    if shape is None:   # fall back to every dim of every tensor input as the problem size
        shape = tuple(d for t in inputs if hasattr(t, "shape") for d in t.shape)

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    key = cache_key(name, bucket_shape(shape), template)
    if not force and key in cache:
        e = cache[key]
        print(f"  cache HIT  [{key}]\n    -> {e['config']}  ({e['time_ms']:.3f} ms, no search)")
        return e["config"]

    # miss: decide how much of the space we can afford to look at
    if len(space) <= budget:
        picks, strategy = list(space), f"grid ({len(space)} configs)"
    else:
        picks = random.Random(seed).sample(space, budget)
        strategy = f"random {budget} of {len(space)}"
    print(f"  cache MISS [{key}] -- strategy: {strategy}")

    GEN.mkdir(exist_ok=True)
    if workers is None:
        workers = os.cpu_count() or 1
    best = None  # (cfg, label, seconds)
    if workers > 1:
        # phase 1: all the compiles at once (cpu subprocess work, threads scale fine)
        nw = min(workers, len(picks))
        t0 = time.perf_counter()
        print(f"  phase 1: compiling {len(picks)} variants, {nw} parallel workers")
        with ThreadPoolExecutor(max_workers=nw) as pool:
            built = list(pool.map(lambda c: _compile_config(name, template, c, run), picks))
        print(f"  phase 2: compiled in {time.perf_counter() - t0:.1f} s -- benchmarking serially")
        # phase 2: the gpu gets each kernel to itself
        for i, (cfg, (label, fn, err)) in enumerate(zip(picks, built), 1):
            if err is None:
                t, ok, err = _measure(fn, inputs, ref, rtol, atol)
            else:
                t, ok = float("inf"), False
            if ok and (best is None or t < best[2]):
                best = (cfg, label, t)
            bs = f"{best[2]*1e3:7.3f}" if best else "    inf"
            note = f"  -- skipped ({err})" if err else ""
            print(f"  [{i:2d}/{len(picks)}] {label:24s} {t*1e3:8.3f} ms   best-so-far {bs} ms{note}")
    else:
        # the original serial path: compile + measure per config, interleaved
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
