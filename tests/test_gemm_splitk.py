"""Correctness tests for the split-K GEMM (GPU-gated)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemm_splitk  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_splitk_matches_fp32() -> None:
    torch.manual_seed(0)
    a = torch.randn(512, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 512, device="cuda", dtype=torch.float16)
    c = gemm_splitk.triton_gemm_splitk(a, b, split_k=4)
    ref = a.float() @ b.float()
    assert torch.allclose(c, ref, atol=1e-2, rtol=1e-2), (
        f"max abs err {(c - ref).abs().max().item():.3e}"
    )


def test_splitk_matches_non_split() -> None:
    import gemm  # noqa: E402

    torch.manual_seed(1)
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    c_split = gemm_splitk.triton_gemm_splitk(a, b, split_k=4)
    c_plain = gemm.triton_gemm(a, b)
    assert torch.allclose(c_split, c_plain, atol=1e-2, rtol=1e-2)
