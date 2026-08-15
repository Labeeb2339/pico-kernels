"""Correctness tests for grouped-query attention (GPU-gated)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import attention_gqa  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hkv", [8, 4, 2, 1])
def test_gqa_matches_eager(dtype, hkv) -> None:
    torch.manual_seed(0)
    Hq, N, D = 8, 256, 64
    q = torch.randn(2, Hq, N, D, device="cuda", dtype=dtype)
    k = torch.randn(2, hkv, N, D, device="cuda", dtype=dtype)
    v = torch.randn(2, hkv, N, D, device="cuda", dtype=dtype)
    o = attention_gqa.triton_attention_gqa(q, k, v)
    ref = attention_gqa.eager_attention_gqa(q, k, v)
    assert torch.allclose(o.float(), ref.float(), atol=5e-2, rtol=5e-2), (
        f"hkv={hkv}: max abs err {(o.float() - ref.float()).abs().max().item():.3e}"
    )


def test_gqa_mha_is_equivalent() -> None:
    # Hq == Hkv (n_repeat=1) must reduce to the MHA kernel's result
    from attention import triton_attention

    torch.manual_seed(1)
    q = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float16)
    o_gqa = attention_gqa.triton_attention_gqa(q, k, v)
    o_mha = triton_attention(q, k, v)
    assert torch.allclose(o_gqa.float(), o_mha.float(), atol=1e-2, rtol=1e-2)
