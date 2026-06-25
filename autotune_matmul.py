"""Autotune a shared-memory tiled matmul: which TILE size is fastest for a given
shape on this GPU? Demonstrates the thesis -- the best config isn't predictable,
and (run a few shapes) it changes with the shape.

    cmd /c "winbuild.bat -m autotune_matmul"     # from the KernelTuner dir
"""

import torch

from tuner import DEVICE, autotune, benchmark

# Tiled matmul C = A@B. One tunable knob: TILE (shared-memory tile edge).
# `__TILE__` is substituted by the tuner (str.replace, so C's own {} are safe).
TEMPLATE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#define TILE __TILE__

__global__ void mm(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int ac = t * TILE + threadIdx.x, br = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && ac < K) ? A[row * K + ac] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (br < K && col < N) ? B[br * N + col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

torch::Tensor run(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    mm<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                        C.data_ptr<float>(), M, N, K);
    return C;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""

CONFIGS = [{"TILE": 8}, {"TILE": 16}, {"TILE": 32}]


def tune_shape(M, K, N):
    A = torch.randn(M, K, device=DEVICE)
    B = torch.randn(K, N, device=DEVICE)
    ref = A @ B
    print(f"\nmatmul {M}x{K} @ {K}x{N}")
    autotune("mm", TEMPLATE, CONFIGS, (A, B), ref)
    t = benchmark(lambda: A @ B)
    print(f"  torch (cuBLAS): {t*1e3:.3f} ms   <-- the unbeatable reference")


if __name__ == "__main__":
    for shape in [(1024, 1024, 1024), (4096, 256, 4096), (256, 4096, 256)]:
        tune_shape(*shape)
