# KernelTuner

A tiny autotuning CUDA kernel optimizer.

**Premise:** the fastest GPU kernel depends on the *shape* and the *hardware*, and you
can't reliably predict it — so don't guess. Generate kernel variants, compile them,
check correctness, benchmark them on the real device, and keep the winner. (The same
reason cuBLAS / cuDNN / Triton autotune.) This is the lesson from the
[TransformerOp](../TransformerOp) project — where a "smarter" attention kernel kept
losing to a simpler one — turned into a reusable tool.

## How it works

1. A kernel **template** with `__KEY__` placeholders (e.g. `__TILE__`).
2. A list of **configs** to sweep (e.g. `TILE ∈ {8, 16, 32}`).
3. `autotune()` substitutes each config, compiles it as a PyTorch CUDA extension,
   gates on `torch.allclose` vs a reference, benchmarks with warmup + `cuda.synchronize`
   + median, and returns the configs fastest-first.

The winner is reported *per shape* — run a few shapes and watch the best config change.

## Run

```powershell
# reuses TransformerOp's venv; winbuild.bat sets up the MSVC build env
cmd /c "winbuild.bat -m autotune_matmul"
```

Requires the same toolchain as TransformerOp Phase 3 (CUDA Toolkit 12.x + MSVC).

## Scope

- **v1 (here):** parameter search over a fixed template (tiled matmul, knob = TILE).
- **Not (yet):** searching over kernel *structures* — that's the hard research part
  (e.g. Mirage's μGraph search with formal equivalence proofs). This tool does the
  tractable, genuinely useful slice: tune a template you wrote against your hardware.
