"""does the generator actually reduce work? three chains:

  1. control: regenerate bias+gelu from a chain spec. the hand-written version
     (autotune_bias_gelu.py) got 0.406ms vs 0.780 eager, the generated one should
     land in the same place.
  2. relu(x + residual), the resnet block tail. never hand-written here, costs one line.
  3. gelu(x + bias) + residual, a 3-op chain. eager pays 2 memory trips per op so the
     fused win should grow with chain length.

judge each fused time against the eager time printed next to it -- absolute numbers
drift between runs, same-run comparisons only.

    cmd /c "winbuild.bat -m fusegen_demo"     # from the KernelTuner dir
"""

import torch

from fusegen import eager_chain, fuse_and_tune
from tuner import DEVICE, benchmark


def show(chain, X, args):
    print(f"\n=== {' -> '.join(chain)} ===")
    fuse_and_tune(chain, X, args)
    t = benchmark(lambda: eager_chain(chain, X, args))
    print(f"  eager chain ({len(chain)} kernels): {t*1e3:.3f} ms")


def main():
    rows, cols = 4096, 4096
    X = torch.randn(rows, cols, device=DEVICE)
    b = torch.randn(cols, device=DEVICE)
    res = torch.randn(rows, cols, device=DEVICE)

    show(["bias_add", "gelu"], X, (b,))                    # control vs the hand-written one
    show(["residual_add", "relu"], X, (res,))              # new fusion, zero new cuda
    show(["bias_add", "gelu", "residual_add"], X, (b, res))  # 3 ops, win should grow


if __name__ == "__main__":
    main()
