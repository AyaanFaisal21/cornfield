"""is parallel compilation actually worth keeping? decide it the way everything else
here gets decided: race it.

both arms run the identical search (sgemm, random 8 of 32, seed 0, same picks).
compile is basically all of the wall time and nvcc/cl are cpu subprocesses, so the
parallel arm should collapse toward the cost of the slowest single compile.

fairness details that matter: torch caches builds by source hash, so whichever arm ran
second would inherit the first arm's binaries and win for free . . . each arm's template
gets its own salt comment (changes the hash, not the code) so both compile all 8 from
scratch. force=True skips the config cache. the parallel arm goes first so a bug in
the newer path fails fast instead of after 8 minutes of serial compiling.

result on this box: 470s serial -> 92s parallel, 5.1x, same winner both arms.

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
