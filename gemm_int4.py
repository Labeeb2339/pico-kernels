"""Packed INT4 GEMM in Triton — the inference speedup behind 4-bit GPTQ.

4-bit weights are packed 2-per-byte, so a quantized linear layer reads 4x fewer
weight bytes than fp16. For MEMORY-BOUND matmuls (small batch, large weight
matrix — the LLM decode regime) that 4x bandwidth saving is the actual speedup.
For COMPUTE-BOUND matmuls the kernel decompresses to fp16 and does the same
tensor-core work plus a bit of unpacking, so it is NOT faster there.

This file shows both regimes honestly: the win is bandwidth, not raw FLOPs.
Weights are packed along N so decompression is a single ``tl.interleave``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.testing


@triton.jit
def int4_gemm_kernel(
    a_ptr, w_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,   # a: (M, K) fp16
    stride_wk, stride_wn,   # w: (K, N//2) uint8 packed
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)          # decompressed N
    offs_pn = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)  # packed N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_pn[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)                        # (BLOCK_M, BLOCK_K) fp16
        w = tl.load(w_ptrs)                        # (BLOCK_K, BLOCK_N//2) uint8
        w_low = (w & 0x0F).to(tl.int32)            # 0..15
        w_high = (w >> 4).to(tl.int32)             # 0..15
        # sign-extend signed int4 (0..15 -> -8..7)
        w_low = tl.where(w_low >= 8, w_low - 16, w_low).to(tl.float16)
        w_high = tl.where(w_high >= 8, w_high - 16, w_high).to(tl.float16)
        w_deq = tl.interleave(w_low, w_high)       # (BLOCK_K, BLOCK_N) fp16
        acc = tl.dot(a, w_deq, acc)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def quantize_int4(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-tensor symmetric int4 quantization of ``w`` (fp16/fp32)."""
    scale = w.abs().max().item() / 7.0
    w_q = torch.round(w / scale).clamp(-8, 7).to(torch.int8)
    return w_q, scale


def pack_int4_n(w_q: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 weights 2-per-byte along N -> (K, N//2) uint8."""
    K, N = w_q.shape
    assert N % 2 == 0
    w = w_q.to(torch.int32).reshape(K, N // 2, 2)
    low = w[:, :, 0] & 0x0F
    high = w[:, :, 1] & 0x0F
    return (low | (high << 4)).to(torch.uint8)


def int4_gemm(a: torch.Tensor, w_packed: torch.Tensor, scale: float,
              BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 64) -> torch.Tensor:
    """``a @ (W_int4 * scale)`` where ``w_packed`` is ``(K, N//2)`` uint8."""
    assert a.is_cuda and a.dtype in (torch.float16, torch.bfloat16)
    M, K = a.shape
    K2, N_half = w_packed.shape
    assert K2 == K and N_half % 1 == 0
    N = N_half * 2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    int4_gemm_kernel[grid](
        a, w_packed, c, M, N, K,
        a.stride(0), a.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c * scale


def check_correctness(M: int = 128, K: int = 256, N: int = 256) -> None:
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(K, N, device="cuda", dtype=torch.float16)
    w_q, scale = quantize_int4(w)
    w_packed = pack_int4_n(w_q)
    c = int4_gemm(a, w_packed, scale)
    ref = a.float() @ (w_q.float() * scale)  # (M,K) @ (K,N) = (M,N)
    rel = torch.norm(c - ref) / torch.norm(ref)
    print(f"int4 vs fp16-quantized ref: rel Frobenius err {rel.item():.4%}")


def bench() -> None:
    print(f"{'shape (M,N,K)':>16} {'fp16':>12} {'int4':>12} {'speedup':>8}  regime")
    print("-" * 60)
    cases = [
        (16, 4096, 4096, "memory-bound (decode)"),
        (16, 8192, 8192, "memory-bound (decode)"),
        (1024, 4096, 4096, "mixed"),
        (4096, 4096, 4096, "compute-bound"),
    ]
    for M, N, K, label in cases:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        w = torch.randn(K, N, device="cuda", dtype=torch.float16)
        w_q, scale = quantize_int4(w)
        w_packed = pack_int4_n(w_q)
        flops = 2 * M * N * K / 1e9

        ms_fp16 = triton.testing.do_bench(lambda: a @ w)
        ms_int4 = triton.testing.do_bench(lambda: int4_gemm(a, w_packed, scale))

        g_fp16 = flops / (ms_fp16 / 1e3)
        g_int4 = flops / (ms_int4 / 1e3)
        print(f"{(M, N, K)!s:>16} {g_fp16:>9.0f} G {g_int4:>9.0f} G "
              f"{g_int4 / g_fp16:>7.2f}x  {label}")


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(0), "| triton", triton.__version__)
    print("=== correctness (int4 vs quantized fp16 ref) ===")
    check_correctness()
    print("=== benchmark (GFLOPS — memory-bound is the win) ===")
    bench()
