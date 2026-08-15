"""Correctness tests for the packed INT4 GEMM (GPU-gated)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemm_int4  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_int4_gemm_correctness() -> None:
    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda", dtype=torch.float16)
    w = torch.randn(256, 512, device="cuda", dtype=torch.float16)
    w_q, scale = gemm_int4.quantize_int4(w)
    w_packed = gemm_int4.pack_int4_n(w_q)
    c = gemm_int4.int4_gemm(a, w_packed, scale)
    ref = a.float() @ (w_q.float() * scale)
    # decompression is exact: the result must match the quantized reference
    assert torch.allclose(c, ref, atol=1e-1, rtol=1e-2), (
        f"rel err {torch.norm(c - ref).item() / torch.norm(ref).item():.4f}"
    )


def test_pack_roundtrip() -> None:
    torch.manual_seed(1)
    w_q = torch.randint(-8, 7, (256, 256), dtype=torch.int8)
    w_packed = gemm_int4.pack_int4_n(w_q)
    low = (w_packed & 0x0F).to(torch.int32)
    high = (w_packed >> 4).to(torch.int32)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    w_recon = torch.stack([low, high], dim=2).reshape(256, 256)
    assert (w_recon == w_q).all(), "pack/unpack round-trip must be lossless"


def test_pack_shape() -> None:
    w_q = torch.randint(-8, 7, (256, 512), dtype=torch.int8)
    w_packed = gemm_int4.pack_int4_n(w_q)
    assert w_packed.shape == (256, 256) and w_packed.dtype == torch.uint8
