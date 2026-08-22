"""Correctness tests for the from-scratch FlashAttention kernel.

GPU-gated (skipped on CPU). Compares the Triton kernel against a fp32 naive
reference and against ``torch.nn.functional.scaled_dot_product_attention``.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import attention  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("N", [128, 256, 512])
def test_attention_matches_naive(dtype: torch.dtype, N: int) -> None:
    torch.manual_seed(0)
    D = 64
    q = torch.randn(2, 4, N, D, device="cuda", dtype=dtype)
    k = torch.randn(2, 4, N, D, device="cuda", dtype=dtype)
    v = torch.randn(2, 4, N, D, device="cuda", dtype=dtype)

    out_tri = attention.triton_attention(q, k, v)
    ref = attention.eager_attention(q, k, v)
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    assert torch.allclose(out_tri.float(), ref.float(), atol=5e-2, rtol=5e-2), (
        f"mismatch at N={N} dtype={dtype}: "
        f"max abs err {(out_tri.float() - ref.float()).abs().max().item():.3e}"
    )
    assert torch.allclose(out_tri.float(), sdpa.float(), atol=5e-2, rtol=5e-2), (
        f"SDPA mismatch at N={N} dtype={dtype}: "
        f"max abs err {(out_tri.float() - sdpa.float()).abs().max().item():.3e}"
    )
    assert out_tri.shape == q.shape
    assert out_tri.dtype == q.dtype
    assert out_tri.device == q.device


def test_attention_causality() -> None:
    """Future keys must not influence earlier queries (causal masking)."""
    torch.manual_seed(0)
    N, D = 64, 32
    B, H = 1, 2
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    out_a = attention.triton_attention(q, k, v)

    # Garbage-ize the *last* key/value. A causal kernel's output for the first
    # N-1 query positions must be unchanged (they never attend to the last key),
    # while the last query (which attends to itself) must change.
    k2 = k.clone()
    k2[:, :, -1, :] = 100.0
    v2 = v.clone()
    v2[:, :, -1, :] = 100.0
    out_b = attention.triton_attention(q, k2, v2)

    assert torch.allclose(
        out_a[:, :, :-1, :].float(), out_b[:, :, :-1, :].float(), atol=1e-2, rtol=1e-2
    ), "earlier query positions changed after mutating a future key"
    assert not torch.allclose(
        out_a[:, :, -1, :].float(), out_b[:, :, -1, :].float(), atol=1e-2, rtol=1e-2
    ), "last query position should depend on the mutated last key"


def test_attention_rejects_noncontiguous_input() -> None:
    q = torch.randn(1, 2, 64, 32, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.float16)
    v = torch.randn_like(q)

    with pytest.raises(ValueError, match="shapes must match"):
        attention.triton_attention(q, k, v)

    k = torch.randn_like(q).transpose(-2, -1).contiguous().transpose(-2, -1)
    assert k.shape == q.shape and not k.is_contiguous()
    with pytest.raises(ValueError, match="k must be contiguous"):
        attention.triton_attention(q, k, v)


def test_attention_rejects_unsupported_shape_and_dtype() -> None:
    q = torch.randn(1, 2, 96, 32, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="N must be divisible"):
        attention.triton_attention(q, q, q)

    q32 = torch.randn(1, 2, 64, 32, device="cuda", dtype=torch.float32)
    with pytest.raises(TypeError, match="fp16 or bf16"):
        attention.triton_attention(q32, q32, q32)

    q_bad_d = torch.randn(1, 2, 64, 48, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="D must be a power of two"):
        attention.triton_attention(q_bad_d, q_bad_d, q_bad_d)

    q_cpu = torch.randn(1, 2, 64, 32, dtype=torch.float16)
    with pytest.raises(ValueError, match="CUDA tensor"):
        attention.triton_attention(q_cpu, q_cpu, q_cpu)
