"""Autotuning harness: generate kernel variants, compile, correctness-check,
benchmark, and pick the winner for the given shape + GPU.

Thesis (learned the hard way in the TransformerOp project): the fastest kernel
depends on shape AND hardware, and you can't reliably predict it -- so generate
candidate kernels, measure them on the real device, and select. That's exactly
why cuBLAS / cuDNN / Triton autotune.
"""

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


def autotune(name, template, configs, inputs, ref, run="run", rtol=1e-3, atol=1e-2):
    """Generate + build + check + benchmark each config; return results fastest-first.

    template : CUDA source using `__KEY__` placeholders (not `{}` -- that clashes with C).
    configs  : list of dicts, e.g. [{"TILE": 8}, {"TILE": 16}]; each `__TILE__` is substituted.
    inputs   : tensors passed to ext.<run>(*inputs).
    ref      : reference output to check correctness against (torch.allclose).
    """
    GEN.mkdir(exist_ok=True)
    results = []
    for cfg in configs:
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
        results.append((label, t, ok))
        flag = "" if ok else "  <-- INCORRECT, skipped"
        print(f"  {label:10s}  {t*1e3:8.3f} ms   correct={ok}{flag}")

    results.sort(key=lambda r: r[1])
    if results and results[0][2]:
        print(f"\n  best correct config: {results[0][0]}  ({results[0][1]*1e3:.3f} ms)")
    return results
