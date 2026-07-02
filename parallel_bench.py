"""v2 Theme B (revised: search THROUGHPUT, not intelligence): is parallel
compilation a meaningful addition? Decide it the project's way -- measure.

Both arms run the IDENTICAL search (sgemm, random 8 of 32, seed 0 -> same 8
configs). Compile is ~95% of trial wall time and nvcc/cl are CPU subprocesses,
so the parallel arm should approach serial_time / min(workers, cpu_cores).

Fairness: torch caches builds by source hash, so whichever arm ran second would
reuse the first arm's binaries and win unfairly. Each arm's template gets a
unique salt comment -- changes the hash, not the code -- forcing both to compile
all 8 variants from scratch. force=True bypasses the config cache. The parallel
arm runs FIRST so a bug in the new path fails fast.

    cmd /c "winbuild.bat -m parallel_bench"     # from the KernelTuner dir
"""

import os
import time

import torch

from autotune_sgemm import TEMPLATE
from autotune_sgemm_search import RANGES, valid
from tuner import DEVICE, make_space, tune


def arm(tag, workers, space, inputs, ref, shape):
    salted = TEMPLATE + f"\n// bench salt: {tag}\n"
    print(f"\n=== {tag} arm (workers={workers}) ===")
    t0 = time.perf_counter()
    cfg = tune("sgemm", salted, space, inputs, ref, shape=shape, budget=8,
               force=True, workers=workers)
    wall = time.perf_counter() - t0
    print(f"  {tag} wall time: {wall:.1f} s")
    return wall, cfg


def main():
    M = K = N = 1024
    A = torch.randn(M, K, device=DEVICE)
    B = torch.randn(K, N, device=DEVICE)
    ref = A @ B
    space = make_space(RANGES, valid)
    workers = os.cpu_count() or 4
    print(f"space: {len(space)} valid configs; search: random 8 (seed 0); cpu_count = {workers}")

    wall_p, cfg_p = arm("parallel", workers, space, (A, B), ref, (M, K, N))
    wall_s, cfg_s = arm("serial", 1, space, (A, B), ref, (M, K, N))

    print(f"\n  serial:   {wall_s:6.1f} s")
    print(f"  parallel: {wall_p:6.1f} s   -> {wall_s / wall_p:.1f}x faster")
    print(f"  same winner: {cfg_p == cfg_s}   (parallel {cfg_p} vs serial {cfg_s})")


if __name__ == "__main__":
    main()
