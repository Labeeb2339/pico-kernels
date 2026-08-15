"""Grouped-Query Attention (GQA) in Triton.

Modern LLMs (Llama, Mistral, Gemma) use GQA: fewer KV heads than query heads,
each KV head shared by a group of query heads. That shrinks the KV cache and the
KV bandwidth for the same model capacity. This is the causal FlashAttention
forward from ``attention.py`` with one change — query head ``h`` reads KV head
``h // n_repeat`` (``n_repeat = H_q / H_kv``).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
import triton.testing

from attention import eager_attention  # MHA reference


@triton.jit
def attn_fwd_gqa(
    Q, K, V, O, sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vk, stride_vn,
    stride_ob, stride_oh, stride_om, stride_on,
    H_Q, H_KV, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    N_REPEAT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_tiles = tl.cdiv(N_CTX, BLOCK_M)
    m_tile = pid % num_m_tiles
    tmp = pid // num_m_tiles
    h = tmp % H_Q
    b = tmp // H_Q
    h_kv = h // N_REPEAT  # grouped: query head h -> KV head h_kv

    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    hi = (m_tile + 1) * BLOCK_M
    for start_n in range(0, hi, BLOCK_N):
        k_ptrs = K + b * stride_kb + h_kv * stride_kh + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
        qk = tl.where(mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v_ptrs = V + b * stride_vb + h_kv * stride_vh + (start_n + offs_n)[:, None] * stride_vk + offs_d[None, :] * stride_vn
        v = tl.load(v_ptrs)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(o_ptrs, acc.to(O.dtype.element_ty))


def triton_attention_gqa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    BLOCK_M: int = 64, BLOCK_N: int = 64,
) -> torch.Tensor:
    """Causal FlashAttention for GQA: ``q (B,Hq,N,D)``, ``k/v (B,Hkv,N,D)``."""
    assert q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
    B, Hq, N, D = q.shape
    Bk, Hkv, Nk, Dk = k.shape
    assert (B, N, D) == (Bk, Nk, Dk) and k.shape == v.shape
    assert Hq % Hkv == 0, "Hq must be a multiple of Hkv"
    n_repeat = Hq // Hkv
    assert N % BLOCK_M == 0 and N % BLOCK_N == 0
    sm_scale = 1.0 / math.sqrt(D)
    o = torch.empty_like(q)
    grid = (B * Hq * triton.cdiv(N, BLOCK_M),)
    attn_fwd_gqa[grid](
        q, k, v, o, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Hq, Hkv, N,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, N_REPEAT=n_repeat,
    )
    return o


def eager_attention_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reference: broadcast KV heads to query heads, then MHA eager attention."""
    n_repeat = q.shape[1] // k.shape[1]
    k_rep = k.repeat_interleave(n_repeat, dim=1)
    v_rep = v.repeat_interleave(n_repeat, dim=1)
    return eager_attention(q, k_rep, v_rep)


def check_correctness(dtype: torch.dtype, Hq: int = 8, Hkv: int = 2, N: int = 512, D: int = 64) -> None:
    torch.manual_seed(0)
    q = torch.randn(2, Hq, N, D, device="cuda", dtype=dtype)
    k = torch.randn(2, Hkv, N, D, device="cuda", dtype=dtype)
    v = torch.randn(2, Hkv, N, D, device="cuda", dtype=dtype)
    o = triton_attention_gqa(q, k, v)
    ref = eager_attention_gqa(q, k, v)
    err = (o.float() - ref.float()).abs().max().item()
    print(f"[{dtype}] GQA Hq={Hq} Hkv={Hkv}: max abs err {err:.3e} | "
          f"allclose {torch.allclose(o.float(), ref.float(), atol=5e-2, rtol=5e-2)}")


def bench() -> None:
    B, Hq, N, D = 2, 8, 2048, 64
    print(f"{'Hkv':>5} {'n_repeat':>9} {'eager GQA':>11} {'triton GQA':>12} {'speedup':>8}")
    print("-" * 48)
    for Hkv in (8, 4, 2, 1):
        q = torch.randn(B, Hq, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, Hkv, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, Hkv, N, D, device="cuda", dtype=torch.float16)
        ms_eager = triton.testing.do_bench(lambda: eager_attention_gqa(q, k, v))
        ms_tri = triton.testing.do_bench(lambda: triton_attention_gqa(q, k, v))
        print(f"{Hkv:>5} {Hq // Hkv:>9} {ms_eager:>10.3f}m {ms_tri:>11.3f}m {ms_eager / ms_tri:>7.2f}x")


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(0), "| triton", triton.__version__)
    print("=== correctness ===")
    check_correctness(torch.float16)
    check_correctness(torch.bfloat16)
    print("=== benchmark (N=2048, Hq=8) ===")
    bench()
