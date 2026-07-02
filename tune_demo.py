"""Demo of tune(), the unified front door (v2 Theme C).

One call does it all: cache lookup -> auto strategy pick -> search -> cache store.
Three behaviors shown here, on the simple tiled-matmul template:

  1. Cache miss, small space  -> strategy picks GRID (space fits the budget).
  2. Near-identical shape     -> shape BUCKETING (1000 -> 1024) turns it into a
                                 cache HIT: no compile, no benchmark, instant.
  3. Cache miss, budget < space -> strategy picks RANDOM sampling.

    cmd /c "winbuild.bat -m tune_demo"     # from the KernelTuner dir
"""

import time

import torch

from autotune_matmul import TEMPLATE
from tuner import DEVICE, make_space, tune

SPACE = make_space({"TILE": [8, 16, 32]})


def demo(M, K, N, budget, note):
    A = torch.randn(M, K, device=DEVICE)
    B = torch.randn(K, N, device=DEVICE)
    ref = A @ B
    print(f"\n=== tune(matmul {M}x{K}x{N}, budget={budget})  -- {note} ===")
    t0 = time.perf_counter()
    cfg = tune("mm", TEMPLATE, SPACE, (A, B), ref, shape=(M, K, N), budget=budget)
    print(f"  -> {cfg}  in {time.perf_counter() - t0:.2f} s")


def main():
    demo(1024, 1024, 1024, 8, "expect MISS, grid (3 configs <= budget 8)")
    demo(1000, 1000, 1000, 8, "expect HIT via bucketing (1000 -> 1024 bucket)")
    demo(256, 4096, 256, 2, "expect MISS, random 2 of 3 (budget < space)")


if __name__ == "__main__":
    main()
