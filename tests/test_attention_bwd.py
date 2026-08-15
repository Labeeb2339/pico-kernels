"""Correctness tests for the fused FlashAttention backward (GPU-gated)."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import attention  # noqa: E402
import attention_bwd  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def _ref_grads(q, k, v, do):
    """Reference dQ/dK/dV from PyTorch autograd of an eager fp32 attention."""
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    N = q.shape[2]
    scale = 1.0 / math.sqrt(q.shape[3])
    s = (qf @ kf.transpose(-2, -1)) * scale
    causal = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
    s = s.masked_fill(~causal, float("-inf"))
    o = torch.softmax(s, dim=-1) @ vf
    (o * do.float()).sum().backward()
    return qf.grad, kf.grad, vf.grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 4, 256, 64), (2, 4, 512, 64)])
def test_backward_matches_autograd(dtype, shape) -> None:
    B, H, N, D = shape
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    do = torch.randn(B, H, N, D, device="cuda", dtype=dtype)

    o = attention.triton_attention(q, k, v)
    dq, dk, dv = attention_bwd.triton_attention_backward(q, k, v, o, do)
    dq_ref, dk_ref, dv_ref = _ref_grads(q, k, v, do)

    for g, g_ref in ((dq, dq_ref), (dk, dk_ref), (dv, dv_ref)):
        assert torch.allclose(g, g_ref, atol=5e-2, rtol=5e-2), (
            f"gradient mismatch: max abs err {(g - g_ref).abs().max().item():.3e}"
        )


def test_backward_grads_are_fp32() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float16)
    do = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float16)
    o = attention.triton_attention(q, k, v)
    dq, dk, dv = attention_bwd.triton_attention_backward(q, k, v, o, do)
    assert dq.dtype == dk.dtype == dv.dtype == torch.float32
    assert dq.shape == dk.shape == dv.shape == q.shape
