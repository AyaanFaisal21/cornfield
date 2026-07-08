"""the fusion generator: stop hand-writing fused elementwise kernels, generate them.

the pitch for fused ops was that libraries can't pre-tune the combinatorial pile of op
combos. but hand-writing one template per combo (autotune_bias_gelu.py, ~90 lines) is
the same non-scaling move, so this makes a fused op cost one line:
fuse_and_tune(["bias_add", "gelu"], X, (b,)).

why elementwise chains specifically: each op is a pure per-element function, so any
chain spliced between one load and one store is correct by construction. no
equivalence proofs needed, which is exactly where general structure search (mirage
etc) gets hard. the eager torch chain doubles as the correctness reference and the
baseline the fused kernel has to beat.

an op registers as: a C expression over a running value v (optionally one tensor arg,
indexed "full" per-element or "col" broadcast per column), a device helper if it needs
one, and its torch equivalent. fuse() splices the chain into the same skeleton the
gelu tuners already proved out (VEC/BLOCK knobs, vectorized fast path, scalar tail,
launch check), and the unchanged tune() engine takes it from there.

v1 limits, honestly: straight chains only (no branches, so no swiglu), fp32, at most
one tensor arg per op.

    cmd /c "winbuild.bat -m fusegen_demo"     # from the KernelTuner dir
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from tuner import make_space, tune


# ---- op registry: one entry = one fusable elementwise op ----

@dataclass(frozen=True)
class Op:
    expr: str                 # C expression over {v} (and {arg} if arg is set)
    torch_fn: object          # eager equivalent: fn(v) or fn(v, arg)
    arg: str = None           # None | "full" (same-shape tensor) | "col" (per-column vector)
    helper: str = ""          # __device__ helper the expression needs (deduped per template)


GELU_HELPER = r"""
__device__ __forceinline__ float gelu_fn(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
"""

OPS = {
    "bias_add":     Op("{v} + {arg}", lambda x, b: x + b, arg="col"),
    "residual_add": Op("{v} + {arg}", lambda x, r: x + r, arg="full"),
    "mul":          Op("{v} * {arg}", lambda x, m: x * m, arg="full"),
    "gelu":         Op("gelu_fn({v})", lambda x: F.gelu(x, approximate="tanh"), helper=GELU_HELPER),
    "relu":         Op("fmaxf({v}, 0.f)", torch.relu),
    "sigmoid":      Op("1.f / (1.f + expf(-{v}))", torch.sigmoid),
}


HEADER = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#define VEC __VEC__
#define BLOCK __BLOCK__
#if VEC == 4
typedef float4 vecT;
#elif VEC == 2
typedef float2 vecT;
#else
typedef float vecT;
#endif
"""


def fuse(chain):
    """assemble a fused CUDA template for `chain` (a list of op names) and return
    (name, template). the chain's expressions run back to back on a running value v
    between ONE load and ONE store -- that's the entire win: an N-op eager chain pays
    2N trips over the data, the fused kernel pays 2."""
    ops = [OPS[c] for c in chain]
    name = "fused_" + "_".join(chain)

    # device helpers, deduped but order kept
    helpers = "".join(dict.fromkeys(op.helper for op in ops if op.helper))

    # walk the chain once. each op contributes one "v = ...;" statement to the
    # vectorized fast path (index base+j) and one to the scalar tail (index j);
    # ops carrying a tensor also add a kernel param, wrapper param, and launch arg
    kparams, wparams, largs, vec_stmts, tail_stmts = [], [], [], [], []
    ai = 0
    for op in ops:
        if op.arg is None:
            vec_stmts.append("v = " + op.expr.format(v="v") + ";")
            tail_stmts.append("v = " + op.expr.format(v="v") + ";")
        else:
            a = f"a{ai}"
            ai += 1
            kparams.append(f"const float* {a}, ")
            wparams.append(f", torch::Tensor t_{a}")
            largs.append(f"t_{a}.data_ptr<float>(), ")
            iv = f"{a}[(base + j) % cols]" if op.arg == "col" else f"{a}[base + j]"
            it = f"{a}[j % cols]" if op.arg == "col" else f"{a}[j]"
            vec_stmts.append("v = " + op.expr.format(v="v", arg=iv) + ";")
            tail_stmts.append("v = " + op.expr.format(v="v", arg=it) + ";")

    src = (HEADER + helpers + r"""
__global__ void fused_kernel(const float* X, """ + "".join(kparams) + r"""float* O, int n, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // vector index
    int base = idx * VEC;
    if (base + VEC <= n) {                              // vectorized fast path
        vecT vin = reinterpret_cast<const vecT*>(X)[idx];
        const float* fi = reinterpret_cast<const float*>(&vin);
        vecT vout;
        float* fo = reinterpret_cast<float*>(&vout);
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float v = fi[j];
            """ + "\n            ".join(vec_stmts) + r"""
            fo[j] = v;
        }
        reinterpret_cast<vecT*>(O)[idx] = vout;
    } else {                                            // scalar tail
        for (int j = base; j < n; ++j) {
            float v = X[j];
            """ + "\n            ".join(tail_stmts) + r"""
            O[j] = v;
        }
    }
}

torch::Tensor run(torch::Tensor X""" + "".join(wparams) + r""") {
    auto Xc = X.contiguous();
    int cols = Xc.size(-1);
    int n = Xc.numel();
    auto O = torch::empty_like(Xc);
    int nvec = (n + VEC - 1) / VEC;
    int blocks = (nvec + BLOCK - 1) / BLOCK;
    fused_kernel<<<blocks, BLOCK>>>(Xc.data_ptr<float>(), """ + "".join(largs) + r"""O.data_ptr<float>(), n, cols);
    cudaError_t e = cudaGetLastError();           // a failed launch is otherwise silent --
    TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));  // and could pass allclose on stale memory
    return O;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
""")
    return name, src


def eager_chain(chain, X, args):
    """run the chain as plain torch, one kernel launch per op. this is both the
    correctness reference and the thing the fused kernel has to beat."""
    v, ai = X, 0
    for c in chain:
        op = OPS[c]
        if op.arg is None:
            v = op.torch_fn(v)
        else:
            v = op.torch_fn(v, args[ai])
            ai += 1
    return v


def fuse_and_tune(chain, X, args=(), **kw):
    """the one-liner: sanity-check the args against the chain, generate the template,
    tune it with the usual elementwise knobs (VEC/BLOCK). returns the best config."""
    need = [OPS[c].arg for c in chain if OPS[c].arg is not None]
    assert len(args) == len(need), f"{chain} needs {len(need)} tensor arg(s), got {len(args)}"
    args = tuple(a.contiguous() for a in args)
    for a, kind in zip(args, need):
        want = X.size(-1) if kind == "col" else X.numel()
        assert a.numel() == want, f"arg has {a.numel()} elements, chain needs {want} ({kind})"

    name, template = fuse(chain)
    ref = eager_chain(chain, X, args)
    space = make_space({"VEC": [1, 2, 4], "BLOCK": [256, 512]})
    return tune(name, template, space, (X, *args), ref, atol=1e-4, **kw)
