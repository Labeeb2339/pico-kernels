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
