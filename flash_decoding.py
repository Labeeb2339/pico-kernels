"""KV cache + FlashDecoding-style split-KV attention for autoregressive decode.

Two ideas that make LLM *generation* fast, distinct from the prefill path in
``attention.py``:

1. **KV cache** — past keys/values are stored, so each new token only computes
   attention against the cached prefix instead of recomputing the whole history.
2. **FlashDecoding** — during decode the query is a *single token* (``M=1``), so
   a standard FlashAttention grid has only one block per head and leaves the GPU
   idle. FlashDecoding instead *splits the KV sequence* across blocks, computes
   per-split ``(max, sum, acc)``, and reduces the partial softmaxes (the same
   online-softmax rescaling used in ``attention.py``, but across splits).

The heavy matmul (``q @ K^T`` over the whole cached prefix) is parallelized; the
softmax reduction is tiny and done in a few torch ops.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
import triton.testing


class KVCache:
    """Append-only K/V cache for autoregressive decoding, shape ``(B, H, N, D)``."""

    def __init__(self, batch: int, heads: int, head_dim: int, max_len: int,
                 device, dtype=torch.float16):
        self.k = torch.empty(batch, heads, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.empty(batch, heads, max_len, head_dim, device=device, dtype=dtype)
        self.pos = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Store one step's ``k, v`` (each ``(B, H, 1, D)``) and return the cached prefix."""
        self.k[:, :, self.pos:self.pos + 1] = k
        self.v[:, :, self.pos:self.pos + 1] = v
        self.pos += 1
        return self.k[:, :, :self.pos], self.v[:, :, :self.pos]

    def reset(self) -> None:
        self.pos = 0


@triton.jit
def flash_decoding_kernel(
    Q, K, V, PARTIALS, sm_scale, H,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_pb, stride_ph, stride_ps,
    split_size,
    HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)   # b * H + h
    s = tl.program_id(1)     # split index
    b = pid // H
    h = pid % H

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + b * stride_qb + h * stride_qh + offs_d)  # (D,)

    n_start = s * split_size
    offs_n = n_start + tl.arange(0, BLOCK_N)

    k = tl.load(K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd).to(tl.float32)
    v = tl.load(V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd).to(tl.float32)

    scores = tl.sum(q[None, :] * k, axis=1) * sm_scale   # (BLOCK_N,)
    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    l = tl.sum(p, axis=0)
    acc = tl.sum(p[:, None] * v, axis=0)                 # (HEAD_DIM,)

    base = PARTIALS + b * stride_pb + h * stride_ph + s * stride_ps
    tl.store(base, m)
    tl.store(base + 1, l)
    tl.store(base + 2 + offs_d, acc)


def flash_decoding(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   split_size: int = 256) -> torch.Tensor:
    """Split-KV attention for a single query ``q`` (``B, H, 1, D``) over ``k, v``.

    Returns ``(B, H, 1, D)``. ``N`` must divide ``split_size``.
    """
    B, H, _, D = q.shape
    _, _, N, _ = k.shape
    assert N % split_size == 0, "N must divide split_size"
    num_splits = N // split_size
    sm_scale = 1.0 / math.sqrt(D)

    partials = torch.empty(B, H, num_splits, 2 + D, device=q.device, dtype=torch.float32)
    flash_decoding_kernel[(B * H, num_splits)](
        q, k, v, partials, sm_scale, H,
        q.stride(0), q.stride(1), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        partials.stride(0), partials.stride(1), partials.stride(2),
        split_size,
        HEAD_DIM=D, BLOCK_N=split_size,
    )

    m = partials[..., 0]           # (B, H, S)
    l = partials[..., 1]           # (B, H, S)
    acc = partials[..., 2:]        # (B, H, S, D)

    m_global = m.max(dim=-1, keepdim=True).values   # (B, H, 1)
    alpha = torch.exp(m - m_global)                 # (B, H, S)
    l_global = (l * alpha).sum(dim=-1, keepdim=True)
    o = (acc * alpha[..., None]).sum(dim=-2) / l_global  # (B, H, D)
    return o[:, :, None, :]


def eager_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Naive reference: single-query attention materializing the full N-vector."""
    B, H, _, D = q.shape
    scale = 1.0 / math.sqrt(D)
    qf, kf, vf = q.float(), k.float(), v.float()
    s = (qf @ kf.transpose(-2, -1)) * scale          # (B, H, 1, N)
    p = torch.softmax(s, dim=-1)
    return (p @ vf).to(q.dtype)


def check_correctness(N: int = 2048, D: int = 64, split_size: int = 256) -> None:
    torch.manual_seed(0)
    B, H = 2, 4
    q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    o = flash_decoding(q, k, v, split_size=split_size)
    ref = eager_decode(q, k, v)
    err = (o.float() - ref.float()).abs().max().item()
    print(f"flash-decoding vs eager (N={N}): max abs err {err:.3e} | "
          f"allclose {torch.allclose(o.float(), ref.float(), atol=1e-2, rtol=1e-2)}")


def bench() -> None:
    B, H, D = 2, 4, 64
    print(f"{'N':>7} {'eager decode':>13} {'flash-decoding':>15} {'speedup':>8}")
    print("-" * 46)
    for N in (1024, 2048, 4096, 8192):
        q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        ms_eager = triton.testing.do_bench(lambda: eager_decode(q, k, v))
        ms_fd = triton.testing.do_bench(lambda: flash_decoding(q, k, v, split_size=256))
        print(f"{N:>7} {ms_eager:>12.3f}m {ms_fd:>12.3f}m {ms_eager / ms_fd:>7.2f}x")


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(0), "| triton", triton.__version__)
    print("=== correctness ===")
    check_correctness()
    print("=== benchmark (decode, M=1) ===")
    bench()
