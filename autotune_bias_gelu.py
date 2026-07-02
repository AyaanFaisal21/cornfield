"""Fused bias+GELU -- the first op here with NO library reference to lose to.

This is the transformer MLP epilogue: GELU(XW + b). Eager torch runs it as TWO
kernels -- add (read X, write tmp), then gelu (read tmp, write out) = 4 full
passes over the data. Fusing them into one kernel is read -> compute -> write =
2 passes. On a memory-bound op that's a structural ~2x that no per-kernel tuning
of the baseline can recover. This is the tuner's real niche: libraries can't
pre-tune the combinatorial space of fused ops, so "generate + measure + select"
is how the best implementation gets found (exactly what Triton/torch.compile do).

Bias indexing: X is row-major (rows, cols), bias is per-column, so flat element
i adds b[i % cols]. The modulo is ALU cost on a memory-bound kernel (free), and
the 16KB bias vector stays L1/L2-resident after first touch.

Uses tune() -- the v2 front door: auto strategy + config cache + shape buckets.

    cmd /c "winbuild.bat -m autotune_bias_gelu"     # from the KernelTuner dir
"""

import torch
from torch.nn import functional as F

from tuner import DEVICE, benchmark, make_space, tune

TEMPLATE = r"""
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

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void bias_gelu_kernel(const float* X, const float* b, float* O, int n, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // vector index
    int base = idx * VEC;
    if (base + VEC <= n) {                              // vectorized fast path
        vecT vin = reinterpret_cast<const vecT*>(X)[idx];
        const float* fi = reinterpret_cast<const float*>(&vin);
        vecT vout;
        float* fo = reinterpret_cast<float*>(&vout);
        #pragma unroll
        for (int j = 0; j < VEC; ++j) fo[j] = gelu(fi[j] + b[(base + j) % cols]);
        reinterpret_cast<vecT*>(O)[idx] = vout;
    } else {                                            // scalar tail
        for (int j = base; j < n; ++j) O[j] = gelu(X[j] + b[j % cols]);
    }
}

torch::Tensor run(torch::Tensor X, torch::Tensor bias) {
    auto Xc = X.contiguous();
    int cols = Xc.size(-1);
    int n = Xc.numel();
    auto O = torch::empty_like(Xc);
    int nvec = (n + VEC - 1) / VEC;
    int blocks = (nvec + BLOCK - 1) / BLOCK;
    bias_gelu_kernel<<<blocks, BLOCK>>>(Xc.data_ptr<float>(), bias.data_ptr<float>(),
                                        O.data_ptr<float>(), n, cols);
    cudaError_t e = cudaGetLastError();           // a failed launch is otherwise silent --
    TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));  // and could pass allclose on stale memory
    return O;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""


def main():
    rows, cols = 4096, 4096                             # MLP-activation-sized (16.7M elements)
    X = torch.randn(rows, cols, device=DEVICE)
    b = torch.randn(cols, device=DEVICE)
    ref = F.gelu(X + b, approximate="tanh")

    space = make_space({"VEC": [1, 2, 4], "BLOCK": [256, 512]})
    print(f"fused bias+gelu, {rows} x {cols}  (space: {len(space)} configs)")
    tune("bias_gelu", TEMPLATE, space, (X, b), ref, shape=(rows, cols), atol=1e-4)

    # the baseline is a CHAIN of library calls, not one optimal call: 2 kernels, 4 memory passes
    t = benchmark(lambda: F.gelu(X + b, approximate="tanh"))
    print(f"  torch eager (add + gelu, 2 kernels): {t*1e3:.3f} ms")


if __name__ == "__main__":
    main()
