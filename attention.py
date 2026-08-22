"""From-scratch FlashAttention (causal) in Triton.

FlashAttention fuses the whole ``softmax(Q K^T / sqrt(d)) V`` into a single
kernel that never materializes the ``N x N`` attention matrix in HBM. That is
what turns attention from O(N^2) memory into O(N) — the difference between
"runs out of memory at 8k tokens" and "runs at 32k".

The kernel is the GEMM structure from ``gemm.py`` plus an **online softmax**:
each ``BLOCK_M x BLOCK_N`` tile of scores is softmax-normalized incrementally
using a running max ``m_i`` and running sum ``l_i``, and the output accumulator
is rescaled whenever the running max updates. The result is bit-comparable to
a two-pass softmax but with a single pass over ``K``/``V``.

Reference: Dao et al., *FlashAttention-2* (2023).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing


# Keep the timing protocol explicit so a benchmark receipt can identify the
# measurement, rather than relying on version-dependent helper defaults.
BENCH_WARMUP_MS = 25
BENCH_REP_MS = 100
BENCH_RETURN_MODE = "median"


# ---------------------------------------------------------------------------
# Kernel (causal forward)
# ---------------------------------------------------------------------------
@triton.jit
def attn_fwd(
    Q, K, V, O, sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vk, stride_vn,
    stride_ob, stride_oh, stride_om, stride_on,
    B, H, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_tiles = tl.cdiv(N_CTX, BLOCK_M)
    m_tile = pid % num_m_tiles
    tmp = pid // num_m_tiles
    h = tmp % H
    b = tmp // H

    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs)  # (BLOCK_M, HEAD_DIM)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Causal: this m-tile can attend to keys up to (m_tile+1)*BLOCK_M.
    hi = (m_tile + 1) * BLOCK_M
    for start_n in range(0, hi, BLOCK_N):
        k_ptrs = K + b * stride_kb + h * stride_kh + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs)  # (BLOCK_N, HEAD_DIM)

        qk = tl.dot(q, tl.trans(k))  # (BLOCK_M, BLOCK_N)
        qk = qk * sm_scale

        # Causal mask: query i can attend to key j only if i >= j.
        mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
        qk = tl.where(mask, qk, float("-inf"))

        # Online-softmax update.
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v_ptrs = V + b * stride_vb + h * stride_vh + (start_n + offs_n)[:, None] * stride_vk + offs_d[None, :] * stride_vn
        v = tl.load(v_ptrs)  # (BLOCK_N, HEAD_DIM)
        acc = tl.dot(p.to(v.dtype), v, acc)  # (BLOCK_M, HEAD_DIM)

        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(o_ptrs, acc.to(O.dtype.element_ty))


def triton_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    BLOCK_M: int = 64, BLOCK_N: int = 64,
) -> torch.Tensor:
    """FlashAttention forward on ``(B, H, N, D)`` tensors, causal, fp16/bf16.

    ``q, k, v`` must be contiguous CUDA tensors with identical ``(B, H, N, D)``
    shapes and fp16/bf16 dtype. This kernel implements causal attention with
    scale ``1 / sqrt(D)`` and no dropout, additive bias, or padding mask.
    """
    tensors = {"q": q, "k": k, "v": v}
    for name, tensor in tensors.items():
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape (B, H, N, D); got {tuple(tensor.shape)}")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(f"{name} must use fp16 or bf16; got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"q/k/v shapes must match; got q={tuple(q.shape)}, "
            f"k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError(f"q/k/v dtypes must match; got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.device != k.device or q.device != v.device:
        raise ValueError(f"q/k/v devices must match; got {q.device}, {k.device}, {v.device}")

    B, H, N, D = q.shape
    if min(B, H, N, D) <= 0:
        raise ValueError(f"B, H, N, and D must be positive; got {tuple(q.shape)}")
    if BLOCK_M <= 0 or BLOCK_M & (BLOCK_M - 1):
        raise ValueError(f"BLOCK_M must be a positive power of two; got {BLOCK_M}")
    if BLOCK_N <= 0 or BLOCK_N & (BLOCK_N - 1):
        raise ValueError(f"BLOCK_N must be a positive power of two; got {BLOCK_N}")
    if N % BLOCK_M != 0 or N % BLOCK_N != 0:
        raise ValueError(
            f"N must be divisible by BLOCK_M and BLOCK_N; got "
            f"N={N}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}"
        )
    if D < 16 or D & (D - 1):
        raise ValueError(f"D must be a power of two and at least 16; got {D}")

    sm_scale = 1.0 / math.sqrt(D)
    o = torch.empty_like(q)
    grid = (B * H * triton.cdiv(N, BLOCK_M),)
    attn_fwd[grid](
        q, k, v, o, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, N,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return o


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
def eager_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Naive O(N^2)-memory attention in fp32 (the mathematically-correct answer)."""
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)
    qf, kf, vf = q.float(), k.float(), v.float()
    s = (qf @ kf.transpose(-2, -1)) * scale  # (B, H, N, N)
    causal = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
    s = s.masked_fill(~causal, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return (p @ vf).to(q.dtype)


# ---------------------------------------------------------------------------
# Correctness + benchmark
# ---------------------------------------------------------------------------
def check_correctness(dtype: torch.dtype, B: int = 4, H: int = 8, N: int = 512, D: int = 64) -> None:
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, D, device="cuda", dtype=dtype)

    out_tri = triton_attention(q, k, v)
    ref = eager_attention(q, k, v)
    err = (out_tri.float() - ref.float()).abs()

    # torch's fused SDPA as a second reference
    out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    print(f"[{dtype}] vs fp32 naive: max abs err {err.max().item():.3e} | "
          f"allclose {torch.allclose(out_tri.float(), ref.float(), atol=5e-2, rtol=5e-2)}")
    print(f"[{dtype}] vs torch SDPA:  max abs err "
          f"{(out_tri.float() - out_sdpa.float()).abs().max().item():.3e}")


def bench() -> None:
    B, H, D = 4, 8, 64
    torch.manual_seed(0)
    print(
        "protocol: fixed provider order eager -> SDPA -> Triton | "
        f"warmup={BENCH_WARMUP_MS}ms rep={BENCH_REP_MS}ms "
        f"stat={BENCH_RETURN_MODE} | seed=0 | B={B} H={H} D={D} fp16"
    )
    print(f"{'N':>6} {'eager':>9} {'SDPA':>9} {'triton':>9} {'vs eager':>9} {'vs SDPA':>9}")
    print("-" * 58)
    for N in (256, 512, 1024, 2048, 4096):
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        ms_eager = triton.testing.do_bench(
            lambda: eager_attention(q, k, v),
            warmup=BENCH_WARMUP_MS,
            rep=BENCH_REP_MS,
            return_mode=BENCH_RETURN_MODE,
        )
        ms_sdpa = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            warmup=BENCH_WARMUP_MS,
            rep=BENCH_REP_MS,
            return_mode=BENCH_RETURN_MODE,
        )
        ms_tri = triton.testing.do_bench(
            lambda: triton_attention(q, k, v),
            warmup=BENCH_WARMUP_MS,
            rep=BENCH_REP_MS,
            return_mode=BENCH_RETURN_MODE,
        )

        print(f"{N:>6} {ms_eager:>8.3f}m {ms_sdpa:>8.3f}m {ms_tri:>8.3f}m "
              f"{ms_eager / ms_tri:>8.2f}x {ms_sdpa / ms_tri:>8.2f}x")

    # 32k: eager would materialize a (4,8,32768,32768) fp32 score matrix = 137 GB,
    # so it OOMs on this GPU. The fused kernels keep O(N) memory and still run.
    N = 32768
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    ms_sdpa = triton.testing.do_bench(
        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
        warmup=BENCH_WARMUP_MS,
        rep=BENCH_REP_MS,
        return_mode=BENCH_RETURN_MODE,
    )
    ms_tri = triton.testing.do_bench(
        lambda: triton_attention(q, k, v),
        warmup=BENCH_WARMUP_MS,
        rep=BENCH_REP_MS,
        return_mode=BENCH_RETURN_MODE,
    )
    print(f"{N:>6} {'OOM':>9} {ms_sdpa:>8.3f}m {ms_tri:>8.3f}m {'-':>9} {ms_sdpa / ms_tri:>8.2f}x")


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(0), "| triton", triton.__version__)
    print("=== correctness ===")
    check_correctness(torch.float16)
    check_correctness(torch.bfloat16)
    print("=== benchmark (ms, lower is better) ===")
    bench()
