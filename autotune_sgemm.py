"""Autotune a register-tiled (2D block-tiling) SGEMM.

Adds the knob that matters: each thread computes a TM x TN grid of outputs, loading
TM+TN operands into registers and reusing them across TM*TN multiply-adds. That
raises arithmetic intensity (compute per memory load) -- the lever that moves a matmul
from shared-memory-bound toward compute-bound, i.e. toward cuBLAS.

Five knobs (BM, BN, BK, TM, TN) -> a big, unpredictable config space, which is exactly
where autotuning pays off. Invalid configs are pruned before compiling.

    cmd /c "winbuild.bat -m autotune_sgemm"     # from the KernelTuner dir
"""

import torch

from tuner import DEVICE, autotune, benchmark

TEMPLATE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#define BM __BM__
#define BN __BN__
#define BK __BK__
#define TM __TM__
#define TN __TN__

__global__ void mm(const float* A, const float* B, float* C, int M, int N, int K) {
    const int numThreads = (BM / TM) * (BN / TN);
    const int tid = threadIdx.x;
    const int threadRow = tid / (BN / TN);     // which TM x TN sub-tile this thread owns
    const int threadCol = tid % (BN / TN);
    const int cRow = blockIdx.y * BM, cCol = blockIdx.x * BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    float acc[TM * TN] = {0.f};
    float regA[TM], regB[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * BK; i += numThreads) {       // cooperatively load A tile
            int r = i / BK, c = i % BK, gr = cRow + r, gc = k0 + c;
            As[i] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        for (int i = tid; i < BK * BN; i += numThreads) {       // cooperatively load B tile
            int r = i / BN, c = i % BN, gr = k0 + r, gc = cCol + c;
            Bs[i] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
        }
        __syncthreads();
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int t = 0; t < TM; ++t) regA[t] = As[(threadRow * TM + t) * BK + k];
            #pragma unroll
            for (int t = 0; t < TN; ++t) regB[t] = Bs[k * BN + threadCol * TN + t];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm)
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn)
                    acc[tm * TN + tn] += regA[tm] * regB[tn];   // reuse regs across TM*TN outputs
        }
        __syncthreads();
    }
    for (int tm = 0; tm < TM; ++tm)
        for (int tn = 0; tn < TN; ++tn) {
            int gr = cRow + threadRow * TM + tm, gc = cCol + threadCol * TN + tn;
            if (gr < M && gc < N) C[gr * N + gc] = acc[tm * TN + tn];
        }
}

torch::Tensor run(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    int numThreads = (BM / TM) * (BN / TN);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    mm<<<grid, numThreads>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                             C.data_ptr<float>(), M, N, K);
    return C;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""

# candidate configs; invalid ones are pruned below before compiling
CANDIDATES = [
    {"BM": 64,  "BN": 64,  "BK": 8,  "TM": 4, "TN": 4},
    {"BM": 64,  "BN": 64,  "BK": 16, "TM": 4, "TN": 4},
    {"BM": 64,  "BN": 64,  "BK": 8,  "TM": 8, "TN": 8},
    {"BM": 128, "BN": 64,  "BK": 8,  "TM": 8, "TN": 4},
    {"BM": 128, "BN": 128, "BK": 8,  "TM": 8, "TN": 8},
    {"BM": 128, "BN": 128, "BK": 16, "TM": 8, "TN": 8},
]


def valid(c):
    threads = (c["BM"] // c["TM"]) * (c["BN"] // c["TN"])
    smem = (c["BM"] * c["BK"] + c["BK"] * c["BN"]) * 4
    return (c["BM"] % c["TM"] == 0 and c["BN"] % c["TN"] == 0
            and threads <= 1024 and smem <= 48 * 1024)


def main():
    M = K = N = 1024
    A = torch.randn(M, K, device=DEVICE)
    B = torch.randn(K, N, device=DEVICE)
    ref = A @ B
    configs = [c for c in CANDIDATES if valid(c)]
    print(f"register-tiled sgemm {M}x{K} @ {K}x{N}  ({len(configs)}/{len(CANDIDATES)} configs valid)")
    autotune("sgemm", TEMPLATE, configs, (A, B), ref)
    print(f"  torch (cuBLAS): {benchmark(lambda: A @ B)*1e3:.3f} ms   <-- reference")
    print("  (v1 simple tiling best was ~3.53 ms on this shape)")


if __name__ == "__main__":
    main()
