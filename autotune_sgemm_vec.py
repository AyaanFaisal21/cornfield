"""v2 / Theme A: register-tiled SGEMM with float4 (128-bit) tile loads.

Only change from autotune_sgemm.py: the two global->shared staging loops load a
float4 (4 contiguous floats) per instruction instead of one float. Fatter loads
use the memory bus better and cut load-instruction count 4x. (Assumes M,N,K are
multiples of the tile dims and BK,BN multiples of 4 -- true for these shapes.)

Compare best-float4 vs the scalar best (0.829 ms from random search) and cuBLAS.

    cmd /c "winbuild.bat -m autotune_sgemm_vec"     # from the KernelTuner dir
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
    const int threadRow = tid / (BN / TN);
    const int threadCol = tid % (BN / TN);
    const int cRow = blockIdx.y * BM, cCol = blockIdx.x * BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    float acc[TM * TN] = {0.f};
    float regA[TM], regB[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // float4 staging loads (4 contiguous floats per instruction)
        for (int i = tid * 4; i < BM * BK; i += numThreads * 4) {
            int r = i / BK, c = i % BK;
            *reinterpret_cast<float4*>(&As[i]) =
                *reinterpret_cast<const float4*>(&A[(cRow + r) * K + (k0 + c)]);
        }
        for (int i = tid * 4; i < BK * BN; i += numThreads * 4) {
            int r = i / BN, c = i % BN;
            *reinterpret_cast<float4*>(&Bs[i]) =
                *reinterpret_cast<const float4*>(&B[(k0 + r) * N + (cCol + c)]);
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
                    acc[tm * TN + tn] += regA[tm] * regB[tn];
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
    cudaError_t e = cudaGetLastError();           // a failed launch is otherwise silent --
    TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));  // and could pass allclose on stale memory
    return C;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""

CONFIGS = [   # 256-thread configs first; the 1024-thread one (register-heavy with float4) last
    {"BM": 128, "BN": 128, "BK": 16, "TM": 8, "TN": 8},
    {"BM": 64,  "BN": 64,  "BK": 16, "TM": 4, "TN": 4},   # grid winner (scalar 0.997)
    {"BM": 128, "BN": 64,  "BK": 8,  "TM": 8, "TN": 4},
    {"BM": 128, "BN": 128, "BK": 8,  "TM": 8, "TN": 8},
    {"BM": 128, "BN": 128, "BK": 8,  "TM": 4, "TN": 4},   # 1024 threads -- random-search winner (scalar 0.829)
]


def main():
    M = K = N = 1024
    A = torch.randn(M, K, device=DEVICE)
    B = torch.randn(K, N, device=DEVICE)
    ref = A @ B
    print(f"float4-load sgemm {M}x{K} @ {K}x{N}")
    autotune("sgemm_vec", TEMPLATE, CONFIGS, (A, B), ref)
    print(f"  torch (cuBLAS): {benchmark(lambda: A @ B)*1e3:.3f} ms")
    print("  (scalar-load best was 0.829 ms)")


if __name__ == "__main__":
    main()
