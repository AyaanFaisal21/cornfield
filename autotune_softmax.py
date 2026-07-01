"""Autotune a row-wise softmax -- a memory-bound *reduction* op, structurally
unlike matmul. Proves the engine is op-agnostic: same autotune/cache machinery,
totally different knobs.

Knobs:
  TPR -- threads cooperating per row (low ~ warp-per-row for short rows; high ~
         block-per-row for long rows). The key, cols-dependent decision.
  RPB -- rows packed per block (occupancy).

3-pass, numerically stable (subtract row max). Reduction across the TPR threads of
each row via shared memory (works for any TPR). Inactive threads still hit every
__syncthreads (no early return) to avoid deadlock.

    cmd /c "winbuild.bat -m autotune_softmax"     # from the KernelTuner dir
"""

import torch
from torch.nn import functional as F

from tuner import DEVICE, autotune, benchmark

TEMPLATE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>
#define TPR __TPR__
#define RPB __RPB__

__global__ void softmax_kernel(const float* X, float* O, int rows, int cols) {
    int tid = threadIdx.x;
    int row_in_block = tid / TPR;
    int lane = tid % TPR;
    int row = blockIdx.x * RPB + row_in_block;
    bool active = row < rows;
    const float* xr = X + (long long)(active ? row : 0) * cols;
    float* orow = O + (long long)(active ? row : 0) * cols;

    __shared__ float sdata[RPB * TPR];
    float* seg = &sdata[row_in_block * TPR];   // this row's reduction scratch

    float m = -FLT_MAX;                         // pass 1: row max
    if (active) for (int c = lane; c < cols; c += TPR) m = fmaxf(m, xr[c]);
    seg[lane] = m; __syncthreads();
    for (int s = TPR / 2; s > 0; s >>= 1) {
        if (lane < s) seg[lane] = fmaxf(seg[lane], seg[lane + s]);
        __syncthreads();
    }
    float row_max = seg[0]; __syncthreads();

    float sum = 0.f;                            // pass 2: sum of exp(x - max)
    if (active) for (int c = lane; c < cols; c += TPR) sum += expf(xr[c] - row_max);
    seg[lane] = sum; __syncthreads();
    for (int s = TPR / 2; s > 0; s >>= 1) {
        if (lane < s) seg[lane] += seg[lane + s];
        __syncthreads();
    }
    float row_sum = seg[0]; __syncthreads();

    float inv = 1.f / row_sum;                  // pass 3: normalize + write
    if (active) for (int c = lane; c < cols; c += TPR) orow[c] = expf(xr[c] - row_max) * inv;
}

torch::Tensor run(torch::Tensor X) {
    auto Xc = X.contiguous();
    int cols = Xc.size(-1);
    int rows = Xc.numel() / cols;
    auto O = torch::empty_like(Xc);
    int threads = RPB * TPR;
    int blocks = (rows + RPB - 1) / RPB;
    softmax_kernel<<<blocks, threads>>>(Xc.data_ptr<float>(), O.data_ptr<float>(), rows, cols);
    cudaError_t e = cudaGetLastError();           // a failed launch is otherwise silent --
    TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));  // and could pass allclose on stale memory
    return O;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""

CONFIGS = [
    {"TPR": 32, "RPB": 1}, {"TPR": 32, "RPB": 4},
    {"TPR": 64, "RPB": 4}, {"TPR": 128, "RPB": 2},
    {"TPR": 128, "RPB": 8}, {"TPR": 256, "RPB": 4},
]


def tune(rows, cols):
    X = torch.randn(rows, cols, device=DEVICE)
    ref = F.softmax(X, dim=-1)
    print(f"\nsoftmax {rows} x {cols}")
    autotune("softmax", TEMPLATE, CONFIGS, (X,), ref, atol=1e-5)
    print(f"  torch: {benchmark(lambda: F.softmax(X, dim=-1))*1e3:.3f} ms")


if __name__ == "__main__":
    tune(98304, 256)    # attention shape: short rows
    tune(4096, 4096)    # wide rows
