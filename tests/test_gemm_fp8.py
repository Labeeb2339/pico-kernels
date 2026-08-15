"""Correctness tests for the from-scratch fp8 GEMM (GPU-gated)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemm_fp8  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_fp8_gemm_accuracy() -> None:
    torch.manual_seed(0)
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    c = gemm_fp8.fp8_gemm(a, b)
    ref = a.float() @ b.float()
    rel = torch.norm(c - ref) / torch.norm(ref)
    # fp8 (e4m3, 3-bit mantissa) is inherently lossy: assert it is
    # "fp8-accurate" (~4% measured), not bit-exact.
    assert rel.item() < 0.10, f"rel Frobenius err {rel.item():.4f} too high for fp8"


def test_fp8_gemm_output_dtype_is_fp32() -> None:
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    c = gemm_fp8.fp8_gemm(a, b)
    assert c.dtype == torch.float32


def test_quantize_roundtrip() -> None:
    torch.manual_seed(1)
    x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    x8, scale = gemm_fp8.quantize(x)
    back = x8.to(torch.float32) / scale
    rel = torch.norm(back - x.float()) / torch.norm(x.float())
    assert rel.item() < 0.05, f"quantize round-trip rel err {rel.item():.4f}"
