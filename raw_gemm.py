"""A raw CUDA C GEMM — one layer below Triton.

No Triton, no cuBLAS, no vendor matmul. The kernel is hand-written CUDA C,
compiled at runtime by the NVRTC compiler that ships inside the torch wheel
(`nvrtc64_120_0.dll`), and launched through the CUDA driver API.

This is deliberately the *simplest correct* tiled GEMM: a 16×16 shared-memory
tile, one output element per thread, no register blocking, no tensor cores.
It exists to make the abstraction layers concrete — the 6× gap to cuBLAS is
exactly what tensor cores and fine register tiling buy you.

Run with ``python raw_gemm.py`` (needs ``cuda-python``, which drives the
bundled NVRTC — no CUDA toolkit install required).
"""

from __future__ import annotations

import torch

try:
    from cuda.core import Program, ProgramOptions, LaunchConfig, Device, launch
    _HAS_CUDA_PYTHON = True
except ImportError:  # pragma: no cover
    _HAS_CUDA_PYTHON = False

CUDA_SRC = r"""
extern "C" __global__ void sgemm(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    float acc = 0.0f;
    int ntiles = K / 16;
    for (int t = 0; t < ntiles; ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * K + t * 16 + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = acc;
}
"""


def build_sgemm(arch: str = "sm_120"):
    """Compile the raw kernel with the bundled NVRTC and return a sync launcher."""
    dev = Device()
    dev.set_current()
    prog = Program(CUDA_SRC, code_type="c++", options=ProgramOptions(arch=arch))
    mod = prog.compile("cubin")
    kern = mod.get_kernel("sgemm")
    stream = dev.create_stream()

    def run(A, B, C, M, N, K):
        grid = ((N + 15) // 16, (M + 15) // 16, 1)
        cfg = LaunchConfig(grid=grid, block=(16, 16, 1))
        launch(stream, cfg, kern, A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K)
        stream.sync()

    return run


def main() -> None:
    import triton.testing

    if not _HAS_CUDA_PYTHON:
        raise SystemExit("cuda-python is required: pip install cuda-python")

    run = build_sgemm()
    print("compiled + loaded raw sgemm for sm_120 (bundled NVRTC)")

    M = N = K = 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    # correctness
    run(A, B, C, M, N, K)
    err = (C - (A @ B)).abs().max().item()
    print(f"correctness: max abs err {err:.4f}")

    ms_cublas = triton.testing.do_bench(lambda: A @ B)
    ms_raw = triton.testing.do_bench(lambda: run(A, B, C, M, N, K))
    flops = 2 * M * N * K
    print("=== benchmark (fp32, 1024^3 - higher is better) ===")
    print(f"cuBLAS fp32:      {flops / (ms_cublas / 1e3) / 1e12:6.1f} TFLOPS")
    print(f"raw CUDA fp32:    {flops / (ms_raw / 1e3) / 1e12:6.1f} TFLOPS")
    print(f"cuBLAS is {ms_raw / ms_cublas:.1f}x faster (tensor cores + register tiling)")


if __name__ == "__main__":
    main()
