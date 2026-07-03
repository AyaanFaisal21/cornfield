# KernelTuner

An autotuning CUDA kernel optimizer.

**Premise:** the fastest GPU kernel depends on the *shape* and the *hardware*, and you
can't reliably predict it — so don't guess. Generate kernel variants, compile them,
check correctness, **benchmark them on the real device**, and keep the winner. (The same
reason cuBLAS / cuDNN / Triton autotune.) It's purely *empirical*: the tuner doesn't model
why a config is fast — it measures wall-clock time, which captures every hardware facet at
once, and picks the fastest. Spun out of the [TransformerOp](../TransformerOp) project.

## What it does

`tuner.py` is op-agnostic: give it a kernel **template** (with `__KEY__` placeholders) and
a **config space**, and it generates, compiles, correctness-gates (`torch.allclose`),
benchmarks (warmup + `cuda.synchronize` + median), and selects.

- **`tune()`** — the front door. One call: cache lookup → auto strategy (grid if the
  space fits the budget, random sampling otherwise) → search → cache the winner. Shapes
  are **bucketed to powers of two**, so a near-identical shape (1000³ after tuning 1024³)
  is an instant cache hit instead of a fresh search. The cache key includes a template
  hash, so editing a kernel auto-invalidates / prevents stale configs.
- **Parallel compilation** — compile is ~95% of a trial's wall time and runs on the CPU,
  so `tune()` compiles all variants in parallel threads (phase 1), then benchmarks them
  serially (phase 2 — GPU timings must never overlap). **Measured 5.1× (470s → 92s) on an
  8-config search, identical winner** (`parallel_bench.py`). `workers=1` keeps the
  original serial path as the archived baseline.
- **`autotune()`** / **`random_search()`** — the bare strategies, callable directly
  (`make_space` builds + prunes the config space).

## Ops covered (one engine, four performance categories)

| op | category | key knobs | best vs torch |
|---|---|---|---|
| matmul (SGEMM) | compute-bound | `BM BN BK TM TN` (tiling, register-blocking) | ~1.5× off cuBLAS |
| softmax | reduction | `TPR RPB` (threads/row, rows/block) | beat on short rows; ~1.1× on wide |
| gelu | elementwise (memory-bound) | `VEC BLOCK` (vectorization width) | ~matched |
| layernorm | fused (reduction + affine) | `TPR RPB` | beat on wide rows; ~matched on short |
| **bias+gelu** | **fused, no library equivalent** | `VEC BLOCK` | **1.9× over eager (2 kernels → 1)** |

bias+gelu is the tuner's real niche: eager torch has *no single op* for it (add kernel +
gelu kernel = 4 memory passes; fused = 2), and libraries can't pre-tune the combinatorial
space of fused ops. Fusion supplies the structural win (even our worst config beat eager);
tuning picks the best config *within* the fused space (a further ~25%, and the winning
vector width differed from plain gelu's — unpredictable as ever).

In every op the optimal config is **shape-dependent and discovered by measurement**: the
matmul tile flips with matrix shape; for the reduction ops (softmax, layernorm) the winner
flips between warp-per-row (`TPR=32`, short rows) and block-per-row (`TPR=256`, long rows);
for gelu the best vector width wasn't the widest. Same engine, totally different knob sets —
it's a *kernel* tuner, not a matmul tuner.

## Run

```powershell
# reuses TransformerOp's venv; winbuild.bat sets up the MSVC build env
& ".\winbuild.bat" -m autotune_sgemm          # register-tiled matmul
& ".\winbuild.bat" -m autotune_sgemm_search   # random search over the config space
& ".\winbuild.bat" -m tune_demo               # unified tune(): auto strategy + cache + shape buckets
& ".\winbuild.bat" -m autotune_softmax        # reduction op
& ".\winbuild.bat" -m autotune_gelu           # elementwise op
& ".\winbuild.bat" -m autotune_layernorm      # fused op
& ".\winbuild.bat" -m autotune_bias_gelu      # fused op with NO library reference (via tune())
```

Requires CUDA Toolkit 12.x + MSVC (same toolchain as TransformerOp Phase 3).

## Scope

- **Done:** empirical tuning over five ops (four categories + a fused op with no library
  equivalent); unified `tune()` API with auto strategy-selection, config caching,
  shape-bucketing, and parallel compilation (5.1× search wall time).
- **Rejected:** a hand-written cost model to pre-rank configs — hard-coding "what makes a
  kernel fast" into the tuner contradicts its premise (performance is hardware-dependent
  and unpredictable: that's why it measures). A cost model *learned from the tuner's own
  per-GPU measurements* stays on the table if parallel random search ever stalls.
- **Not (yet):** searching kernel *structures* rather than template params — the research
  frontier (e.g. Mirage's μGraph search with formal equivalence proofs). This tool does the
  tractable, useful slice: tune templates you wrote against your hardware.
