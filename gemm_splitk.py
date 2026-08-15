"""Split-K GEMM in Triton — when splitting the K-reduction across blocks helps.

A standard GEMM grid is `(M/BLOCK_M, N/BLOCK_N)`; each block reduces the *full*
K. If `M x N` is small there are few blocks and the GPU is underutilized, so
the K-reduction becomes the serial bottleneck. Split-K adds a third grid axis:
each block reduces a *slice* of K, and the partial sums are reduced with an
atomic add into the output.

The honest finding (RTX 5070 Laptop, Triton 3.7.1): a *naive* split-K with
fp32 atomics is **slower** than the plain autotuned kernel at every shape
tested — atomic reduction contention plus a fixed (non-autotuned) tile shape
eats the extra parallelism. Split-K only pays off when the grid is too small to
hide K-reduction latency *and* the reduction is done without atomic contention
(a two-pass partials-buffer + reduce kernel). Kept here as a documented "useful
failure case": correct, but not a win at these shapes on this GPU.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.testing


@triton.jit
def gemm_splitk_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    k_per_split = K // SPLIT_K
    k_start = pid_k * k_per_split

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak
    b_ptrs = b_ptr + (k_start + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k in range(0, k_per_split, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.atomic_add(c_ptrs, acc)


def triton_gemm_splitk(a: torch.Tensor, b: torch.Tensor, split_k: int = 4,
                       BLOCK_M: int = 128, BLOCK_N: int = 128, BLOCK_K: int = 64) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and K % (split_k * BLOCK_K) == 0, "K must divide split_k * BLOCK_K"
    c = torch.zeros((M, N), device=a.device, dtype=torch.float32)  # zero for atomics
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), split_k)
    gemm_splitk_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SPLIT_K=split_k, GROUP_M=8,
    )
    return c


def check_correctness(M: int = 1024, K: int = 1024, N: int = 1024) -> None:
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = triton_gemm_splitk(a, b, split_k=4)
    ref = a.float() @ b.float()
    print(f"split-K vs fp32: max abs err {(c - ref).abs().max().item():.3e} | "
          f"allclose {torch.allclose(c, ref, atol=1e-2, rtol=1e-2)}")


def bench() -> None:
    import gemm  # noqa: F401 - non-split autotuned kernel for comparison

    print(f"{'shape (M,N,K)':>16} {'cuBLAS':>10} {'autotuned':>10} {'split-K':>10}  note")
    print("-" * 64)
    cases = [
        (512, 512, 8192, 4, "tall-skinny"),
        (512, 512, 16384, 8, "tall-skinny"),
        (1024, 1024, 4096, 4, "square"),
        (4096, 4096, 4096, 4, "square"),
    ]
    for M, N, K, sk, note in cases:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        flops = 2 * M * N * K / 1e9
        ms_cublas = triton.testing.do_bench(lambda: a @ b)
        ms_auto = triton.testing.do_bench(lambda: gemm.triton_gemm(a, b))
        ms_split = triton.testing.do_bench(lambda: triton_gemm_splitk(a, b, split_k=sk))
        fmt = lambda ms: f"{flops / (ms / 1e3):>7.0f} G"
        print(f"{(M, N, K)!s:>16} {fmt(ms_cublas):>10} {fmt(ms_auto):>10} {fmt(ms_split):>10}  {note}")


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(0), "| triton", triton.__version__)
    print("=== correctness ===")
    check_correctness()
    print("=== benchmark (GFLOPS) ===")
    bench()
