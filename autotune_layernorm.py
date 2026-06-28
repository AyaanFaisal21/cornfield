"""Autotune LayerNorm -- a FUSED op: a reduction (mean + variance over each row) AND
an elementwise affine (normalize, scale, shift) in one kernel. Reduction-shaped, so it
reuses softmax's knobs (TPR = threads per row, RPB = rows per block), but it fuses the
reduction with the elementwise pass -- the "fused" category, where custom kernels earn
their keep by not re-reading the data.

Fourth op category (compute-bound / reduction / elementwise / fused) -- same engine.

    cmd /c "winbuild.bat -m autotune_layernorm"     # from the KernelTuner dir
"""

import torch
from torch.nn import functional as F

from tuner import DEVICE, autotune, benchmark

TEMPLATE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#define TPR __TPR__
#define RPB __RPB__

__global__ void ln_kernel(const float* X, const float* gamma, const float* beta,
                          float* O, int rows, int cols, float eps) {
    int tid = threadIdx.x;
    int rib = tid / TPR;
    int lane = tid % TPR;
    int row = blockIdx.x * RPB + rib;
    bool active = row < rows;
    const float* xr = X + (long long)(active ? row : 0) * cols;
    float* orow = O + (long long)(active ? row : 0) * cols;

    __shared__ float ss[RPB * TPR];     // sum
    __shared__ float sq[RPB * TPR];     // sum of squares
    float* seg_s = &ss[rib * TPR];
    float* seg_q = &sq[rib * TPR];

    float s = 0.f, q = 0.f;             // one pass: accumulate sum and sumsq together
    if (active) for (int c = lane; c < cols; c += TPR) { float v = xr[c]; s += v; q += v * v; }
    seg_s[lane] = s; seg_q[lane] = q; __syncthreads();
    for (int st = TPR / 2; st > 0; st >>= 1) {
        if (lane < st) { seg_s[lane] += seg_s[lane + st]; seg_q[lane] += seg_q[lane + st]; }
        __syncthreads();
    }
    float mean = seg_s[0] / cols;
    float rstd = rsqrtf(seg_q[0] / cols - mean * mean + eps);
    __syncthreads();

    if (active) for (int c = lane; c < cols; c += TPR)   // fused normalize + affine
        orow[c] = (xr[c] - mean) * rstd * gamma[c] + beta[c];
}

torch::Tensor run(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta) {
    auto Xc = X.contiguous();
    int cols = Xc.size(-1), rows = Xc.numel() / cols;
    auto O = torch::empty_like(Xc);
    int threads = RPB * TPR, blocks = (rows + RPB - 1) / RPB;
    ln_kernel<<<blocks, threads>>>(Xc.data_ptr<float>(), gamma.data_ptr<float>(),
                                   beta.data_ptr<float>(), O.data_ptr<float>(), rows, cols, 1e-5f);
    return O;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""

CONFIGS = [{"TPR": t, "RPB": r} for t in (32, 128, 256) for r in (1, 4)]


def tune(rows, cols):
    X = torch.randn(rows, cols, device=DEVICE)
    g = torch.randn(cols, device=DEVICE)
    b = torch.randn(cols, device=DEVICE)
    ref = F.layer_norm(X, (cols,), g, b)
    print(f"\nlayernorm {rows} x {cols}")
    autotune("ln", TEMPLATE, CONFIGS, (X, g, b), ref, rtol=1e-3, atol=1e-3)
    print(f"  torch: {benchmark(lambda: F.layer_norm(X, (cols,), g, b))*1e3:.3f} ms")


if __name__ == "__main__":
    tune(98304, 256)
    tune(4096, 4096)
