"""random search over the sgemm space. grid compiles everything, fine for a handful of
configs, hopeless for hundreds; this samples a budget from the full pruned space
instead. watch best-so-far converge: the first run found the eventual winner on trial
1, good configs are common in this space.

    cmd /c "winbuild.bat -m autotune_sgemm_search"     # from the KernelTuner dir
"""

import torch

from autotune_sgemm import TEMPLATE          # reuse the register-tiled sgemm template
from tuner import DEVICE, benchmark, make_space, random_search

# knob ranges -> the space (cartesian product, then pruned by valid)
RANGES = {"BM": [64, 128], "BN": [64, 128], "BK": [8, 16], "TM": [4, 8], "TN": [4, 8]}


def valid(c):
    # same launch limits as autotune_sgemm.py
    threads = (c["BM"] // c["TM"]) * (c["BN"] // c["TN"])
    smem = (c["BM"] * c["BK"] + c["BK"] * c["BN"]) * 4
    return (c["BM"] % c["TM"] == 0 and c["BN"] % c["TN"] == 0
            and threads <= 1024 and smem <= 48 * 1024)


def main():
    M = K = N = 1024
    A = torch.randn(M, K, device=DEVICE)
    B = torch.randn(K, N, device=DEVICE)
    ref = A @ B

    space = make_space(RANGES, valid)
    print(f"sgemm {M}x{K} @ {K}x{N}: full valid space = {len(space)} configs")
    random_search("sgemm", TEMPLATE, space, (A, B), ref, budget=8, seed=0)
    print(f"  torch (cuBLAS): {benchmark(lambda: A @ B)*1e3:.3f} ms   <-- reference")


if __name__ == "__main__":
    main()
