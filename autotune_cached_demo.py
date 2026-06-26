"""Config cache demo: tune once, then look up instantly.

First call (miss) runs the search and stores the winner keyed by (op, shape, gpu,
template). Second call (hit) skips the search entirely and returns the stored config.

    cmd /c "winbuild.bat -m autotune_cached_demo"     # from the KernelTuner dir
"""

import time

import torch

from autotune_sgemm import TEMPLATE
from autotune_sgemm_search import RANGES, valid
from tuner import DEVICE, autotune_cached, make_space


def main():
    M = K = N = 1024
    A = torch.randn(M, K, device=DEVICE)
    B = torch.randn(K, N, device=DEVICE)
    ref = A @ B
    space = make_space(RANGES, valid)
    shape = (M, K, N)

    print("=== first call: force a fresh search (cache miss) ===")
    t0 = time.perf_counter()
    cfg1 = autotune_cached("sgemm", TEMPLATE, space, (A, B), ref, shape, budget=8, force=True)
    t_miss = time.perf_counter() - t0

    print("\n=== second call: cache hit (no search) ===")
    t0 = time.perf_counter()
    cfg2 = autotune_cached("sgemm", TEMPLATE, space, (A, B), ref, shape, budget=8)
    t_hit = time.perf_counter() - t0

    assert cfg1 == cfg2, "cache returned a different config!"
    print(f"\n  search (miss): {t_miss:7.2f} s")
    print(f"  lookup (hit):  {t_hit:7.4f} s   -> {t_miss/max(t_hit,1e-9):,.0f}x faster, same config")


if __name__ == "__main__":
    main()
