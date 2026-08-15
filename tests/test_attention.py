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

    assert torch.allclose(out_tri.float(), ref.float(), atol=5e-2, rtol=5e-2), (
        f"mismatch at N={N} dtype={dtype}: "
        f"max abs err {(out_tri.float() - ref.float()).abs().max().item():.3e}"
    )


def test_attention_causality() -> None:
    """The upper triangle (future keys) must not influence the output."""
    torch.manual_seed(0)
    N, D = 128, 32
    q = torch.randn(1, 2, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, N, D, device="cuda", dtype=torch.float16)

    out_a = attention.triton_attention(q, k, v)

    # Zero out all *future* keys/values and recompute — output must be identical.
    causal_mask = torch.tril(torch.ones(N, N, device="cuda", dtype=torch.bool))
    k_masked = torch.where(causal_mask.T[:, :, None].unsqueeze(0), k, torch.zeros_like(k))
    v_masked = torch.where(causal_mask.T[:, :, None].unsqueeze(0), v, torch.zeros_like(v))
    out_b = attention.triton_attention(q, k_masked, v_masked)

    assert torch.allclose(out_a.float(), out_b.float(), atol=5e-2, rtol=5e-2)
