"""demo of tune(), the one-call entry point, against the simple tiled matmul:

  1. fresh shape, small space -> cache miss, picks grid on its own (3 configs <= budget 8)
  2. 1000^3 right after tuning 1024^3 -> shape bucketing turns it into a HIT: no
     compiles, no benchmarks, instant config
  3. budget 2 against 3 configs -> miss, picks random sampling on its own

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
