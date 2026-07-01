"""Autotune GELU -- a pure ELEMENTWISE op (memory-bound, no reduction). Each output
depends only on its own input, so it's embarrassingly parallel and bandwidth-limited.
The key knob is therefore VECTORIZATION: load VEC floats per instruction (float4 =
128-bit transactions) to use memory bandwidth well. BLOCK tunes occupancy.

A third op category for the tuner (compute-bound matmul, reduction softmax, now
memory-bound elementwise) -- same engine, yet another knob set.

    cmd /c "winbuild.bat -m autotune_gelu"     # from the KernelTuner dir
"""

import torch
from torch.nn import functional as F

from tuner import DEVICE, autotune, benchmark

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

__global__ void gelu_kernel(const float* X, float* O, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // vector index
    int base = idx * VEC;
    if (base + VEC <= n) {                              // vectorized fast path (128-bit if VEC=4)
        vecT vin = reinterpret_cast<const vecT*>(X)[idx];
        const float* fi = reinterpret_cast<const float*>(&vin);
        vecT vout;
        float* fo = reinterpret_cast<float*>(&vout);
        #pragma unroll
        for (int j = 0; j < VEC; ++j) fo[j] = gelu(fi[j]);
        reinterpret_cast<vecT*>(O)[idx] = vout;
    } else {                                            // scalar tail (n not a multiple of VEC)
        for (int j = base; j < n; ++j) O[j] = gelu(X[j]);
    }
}

torch::Tensor run(torch::Tensor X) {
    auto Xc = X.contiguous();
    int n = Xc.numel();
    auto O = torch::empty_like(Xc);
    int nvec = (n + VEC - 1) / VEC;
    int blocks = (nvec + BLOCK - 1) / BLOCK;
    gelu_kernel<<<blocks, BLOCK>>>(Xc.data_ptr<float>(), O.data_ptr<float>(), n);
    cudaError_t e = cudaGetLastError();           // a failed launch is otherwise silent --
    TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));  // and could pass allclose on stale memory
    return O;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""

CONFIGS = [{"VEC": v, "BLOCK": b} for v in (1, 2, 4) for b in (256, 512)]


def main():
    X = torch.randn(4096, 4096, device=DEVICE)          # 16.7M elements
    ref = F.gelu(X, approximate="tanh")
    print(f"gelu, {X.numel():,} elements")
    autotune("gelu", TEMPLATE, CONFIGS, (X,), ref, atol=1e-4)
    print(f"  torch: {benchmark(lambda: F.gelu(X, approximate='tanh'))*1e3:.3f} ms")


if __name__ == "__main__":
    main()
