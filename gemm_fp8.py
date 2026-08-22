"""From-scratch fp8 GEMM (e4m3) in Triton — Blackwell tensor cores.

Blackwell (sm_120) adds native fp8 (e4m3/e5m2) tensor cores that run at ~2x the
fp16 rate. This kernel is a *scaled* fp8 matmul:

1. inputs are cast to e4m3 with per-tensor scale factors so the data uses the
   full fp8 dynamic range (e4m3 max from ``torch.finfo``, not hard-coded),
2. the tensor-core matmul runs in fp8 with an fp32 accumulator,
3. the output is rescaled once in the epilogue.

The cast/scale is an O(n^2) memory pass done *once* (offline for weights, once
per activation in real inference); the O(n^3) matmul is what is benchmarked.
Accuracy loss vs fp16 is the whole point of the trade, so it is measured and
reported rather than hidden.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.testing

_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)

_CONFIGS: list[tuple[int, int, int, int, int, int]] = [
    (64, 64, 32, 8, 4, 3),
    (64, 64, 64, 8, 4, 4),
    (64, 128, 64, 8, 4, 3),
    (128, 64, 64, 8, 4, 4),
    (128, 128, 32, 8, 8, 3),
    (128, 128, 64, 8, 8, 3),
    (128, 256, 64, 8, 8, 3),
    (256, 128, 64, 8, 8, 3),
]


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm},
            num_warps=nw, num_stages=ns,
        )
        for (bm, bn, bk, gm, nw, ns) in _CONFIGS
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fp8_gemm_kernel(
    a_ptr, b_ptr, c_ptr, out_scale,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
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

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)          # fp8 e4m3
        b = tl.load(b_ptrs)          # fp8 e4m3
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc * out_scale)


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Cast ``x`` to e4m3, scaled so ``max|x|`` maps to the e4m3 max.

    Returns ``(x_fp8, scale)`` where ``x ≈ x_fp8 / scale``.
    """
    scale = _E4M3_MAX / x.abs().max().item()
    return (x * scale).to(torch.float8_e4m3fn), scale


def fp8_gemm_precast(
    a8: torch.Tensor, a_scale: float,
    b8: torch.Tensor, b_scale: float,
) -> torch.Tensor:
    """Kernel-only fp8 matmul on already-cast ``a8, b8`` (fp32 output)."""
    M, K = a8.shape
    K2, N = b8.shape
    assert K == K2
    out_scale = 1.0 / (a_scale * b_scale)
    c = torch.empty((M, N), device=a8.device, dtype=torch.float32)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    fp8_gemm_kernel[grid](
        a8, b8, c, out_scale, M, N, K,
        a8.stride(0), a8.stride(1),
        b8.stride(0), b8.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def fp8_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Convenience: quantize then matmul (the full fp8 path)."""
    a8, a_scale = quantize(a)
    b8, b_scale = quantize(b)
    return fp8_gemm_precast(a8, a_scale, b8, b_scale)


def check_correctness(M: int = 1024, K: int = 1024, N: int = 1024) -> None:
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = fp8_gemm(a, b)
    ref = a.float() @ b.float()
    rel_fro = torch.norm(c - ref) / torch.norm(ref)
    print(f"fp8 vs fp32: rel Frobenius err {rel_fro.item():.4%} | "
          f"max abs err {(c - ref).abs().max().item():.3e}")


def bench() -> None:
    print(f"{'size':>7} {'fp16 cuBLAS':>12} {'fp8 triton':>12} {'speedup':>8}")
    print("-" * 44)
    for size in (512, 1024, 2048, 4096):
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)
        a8, a_scale = quantize(a)   # O(n^2), done once / offline — not timed
        b8, b_scale = quantize(b)
        flops = 2 * size**3 / 1e9  # GFLOP

        ms_fp16 = triton.testing.do_bench(lambda: a @ b)
        ms_fp8 = triton.testing.do_bench(
            lambda: fp8_gemm_precast(a8, a_scale, b8, b_scale)
        )

        g_fp16 = flops / (ms_fp16 / 1e3)
        g_fp8 = flops / (ms_fp8 / 1e3)
        print(f"{size:>7} {g_fp16:>9.0f} G {g_fp8:>9.0f} G {g_fp8 / g_fp16:>7.2f}x")


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(0), "| triton", triton.__version__)
    print("=== correctness (fp8 vs fp32 reference) ===")
    check_correctness()
    print("=== benchmark (GFLOPS - higher is better; cast is pre-done) ===")
    bench()
