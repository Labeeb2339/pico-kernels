"""From-scratch FlashAttention **backward pass** in Triton (causal).

The forward (``attention.py``) fuses ``softmax(QK^T/√d)V`` into one kernel. The
backward computes ``dQ, dK, dV`` given ``dO`` — also fused, and also without
ever materializing the ``N x N`` score matrix.

Algorithm (FlashAttention-2, "recompute"): the softmax statistics ``m`` (row
max) and ``l`` (row sum) are recomputed in a first pass, then a second pass
uses ``P = exp(S - m)/l`` to accumulate

* ``dV = P^T @ dO``          (atomic — each key tile gathers from many query tiles)
* ``dP = dO @ V^T``
* ``dS = P * (dP - D)`` with ``D = rowsum(dO * O)``   (softmax Jacobian)
* ``dQ += dS @ K``, ``dK += dS^T @ Q``  (atomic for dK)

Gradients are computed and accumulated in fp32, so they match a fp32 reference
to fp16-forward tolerance (~1e-2). Correctness is verified against PyTorch
autograd of an eager attention, not against a hand-derived formula.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing


@triton.jit
def attn_bwd_kernel(
    Q, K, V, O, dO, dQ, dK, dV, sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vk, stride_vn,
    stride_ob, stride_oh, stride_om, stride_on,
    stride_dob, stride_doh, stride_dom, stride_don,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_dvb, stride_dvh, stride_dvk, stride_dvn,
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

    q = tl.load(Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk).to(tl.float32)
    o = tl.load(O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on).to(tl.float32)
    do = tl.load(dO + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_don).to(tl.float32)

    # D = rowsum(dO * O) — the softmax-backward diagonal term.
    D = tl.sum(do * o, axis=1)  # (BLOCK_M,)

    hi = (m_tile + 1) * BLOCK_M  # causal: this m-tile attends to keys < hi

    # Pass 1: recompute running max m_i and running sum l_i (online softmax).
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start_n in range(0, hi, BLOCK_N):
        k = tl.load(K + b * stride_kb + h * stride_kh + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk).to(tl.float32)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], s, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Pass 2: dV (atomic), dP -> dS, dQ, dK (atomic).
    dq = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    for start_n in range(0, hi, BLOCK_N):
        k = tl.load(K + b * stride_kb + h * stride_kh + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk).to(tl.float32)
        v = tl.load(V + b * stride_vb + h * stride_vh + (start_n + offs_n)[:, None] * stride_vk + offs_d[None, :] * stride_vn).to(tl.float32)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], s, float("-inf"))
        p = tl.exp(s - m_i[:, None]) / l_i[:, None]  # (BLOCK_M, BLOCK_N)

        dv_tile = tl.dot(tl.trans(p), do)  # (BLOCK_N, HEAD_DIM)
        tl.atomic_add(dV + b * stride_dvb + h * stride_dvh + (start_n + offs_n)[:, None] * stride_dvk + offs_d[None, :] * stride_dvn, dv_tile)

        dp = tl.dot(do, tl.trans(v))                    # (BLOCK_M, BLOCK_N)
        ds = p * (dp - D[:, None])                      # softmax Jacobian
        dq += tl.dot(ds, k) * sm_scale                  # (BLOCK_M, HEAD_DIM); S = QK^T * sm_scale

        dk_tile = tl.dot(tl.trans(ds), q) * sm_scale    # (BLOCK_N, HEAD_DIM)
        tl.atomic_add(dK + b * stride_dkb + h * stride_dkh + (start_n + offs_n)[:, None] * stride_dkn + offs_d[None, :] * stride_dkk, dk_tile)

    tl.store(dQ + b * stride_dqb + h * stride_dqh + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqk, dq)


def triton_attention_backward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    o: torch.Tensor, do: torch.Tensor,
    BLOCK_M: int = 64, BLOCK_N: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused causal attention backward. Returns ``(dQ, dK, dV)`` in fp32.

    ``q, k, v, o, do`` must be contiguous ``(B, H, N, D)`` tensors (fp16/bf16
    is fine; gradients accumulate in fp32 and are returned in fp32).
    """
    assert q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
    assert q.shape == k.shape == v.shape == o.shape == do.shape
    B, H, N, D = q.shape
    assert N % BLOCK_M == 0 and N % BLOCK_N == 0
    sm_scale = 1.0 / math.sqrt(D)

    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    grid = (B * H * triton.cdiv(N, BLOCK_M),)
    attn_bwd_kernel[grid](
        q, k, v, o, do, dq, dk, dv, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        B, H, N,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return dq, dk, dv


def _eager_attention_fp32(q, k, v, do) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference: eager fp32 attention with autograd (returns grads)."""
    q = q.detach().float().requires_grad_(True)
    k = k.detach().float().requires_grad_(True)
    v = v.detach().float().requires_grad_(True)
    N = q.shape[2]
    scale = 1.0 / math.sqrt(q.shape[3])
    s = (q @ k.transpose(-2, -1)) * scale
    causal = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
    s = s.masked_fill(~causal, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = p @ v
    (o * do.detach().float()).sum().backward()
    return q.grad, k.grad, v.grad


def check_correctness(dtype: torch.dtype, B: int = 2, H: int = 4, N: int = 256, D: int = 64) -> None:
    from attention import triton_attention  # local import (same package)

    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    do = torch.randn(B, H, N, D, device="cuda", dtype=dtype)

    o = triton_attention(q, k, v)
    dq, dk, dv = triton_attention_backward(q, k, v, o, do)
    dq_ref, dk_ref, dv_ref = _eager_attention_fp32(q, k, v, do)

    for name, g, g_ref in (("dQ", dq, dq_ref), ("dK", dk, dk_ref), ("dV", dv, dv_ref)):
        err = (g - g_ref).abs().max().item()
        ok = torch.allclose(g, g_ref, atol=5e-2, rtol=5e-2)
        print(f"[{dtype}] {name}: max abs err {err:.3e} | allclose {ok}")


def bench() -> None:
    from attention import triton_attention, eager_attention

    B, H, D = 2, 4, 64
    print(f"{'N':>6} {'eager fwd+bwd':>14} {'triton fwd+bwd':>14} {'speedup':>8}")
    print("-" * 46)
    for N in (256, 512, 1024, 2048):
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        do = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        def eager():
            qf = q.detach().float().requires_grad_(True)
            kf = k.detach().float().requires_grad_(True)
            vf = v.detach().float().requires_grad_(True)
            scale = 1.0 / math.sqrt(D)
            s = (qf @ kf.transpose(-2, -1)) * scale
            causal = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
            s = s.masked_fill(~causal, float("-inf"))
            o = torch.softmax(s, dim=-1) @ vf
            (o * do.float()).sum().backward()
            return qf.grad, kf.grad, vf.grad

        def fused():
            o = triton_attention(q, k, v)
            return triton_attention_backward(q, k, v, o, do)

        ms_eager = triton.testing.do_bench(eager)
        ms_tri = triton.testing.do_bench(fused)
        print(f"{N:>6} {ms_eager:>12.3f}m {ms_tri:>12.3f}m {ms_eager / ms_tri:>7.2f}x")


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(0), "| triton", triton.__version__)
    print("=== correctness (vs torch autograd) ===")
    check_correctness(torch.float16)
    check_correctness(torch.bfloat16)
    print("=== benchmark (fwd+bwd, ms) ===")
    bench()
