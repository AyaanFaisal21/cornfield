"""Random search over the register-tiled SGEMM config space.

Grid search compiles every config -- fine for a handful, hopeless for hundreds.
Random search samples a budget from the full (auto-generated, validity-pruned)
space and finds a near-best config in a fraction of the compiles. Watch the
"best-so-far" converge.

    cmd /c "winbuild.bat -m autotune_sgemm_search"     # from the KernelTuner dir
"""

import torch

from autotune_sgemm import TEMPLATE          # reuse the register-tiled SGEMM template
from tuner import DEVICE, benchmark, make_space, random_search

# knob ranges -> the search space (cartesian product, then pruned)
RANGES = {"BM": [64, 128], "BN": [64, 128], "BK": [8, 16], "TM": [4, 8], "TN": [4, 8]}


def valid(c):
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
