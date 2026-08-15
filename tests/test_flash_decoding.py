"""Correctness tests for KV cache + FlashDecoding (GPU-gated)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flash_decoding  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_flash_decoding_matches_eager() -> None:
    torch.manual_seed(0)
    B, H, N, D = 2, 4, 1024, 64
    q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    o = flash_decoding.flash_decoding(q, k, v, split_size=256)
    ref = flash_decoding.eager_decode(q, k, v)
    assert torch.allclose(o.float(), ref.float(), atol=1e-2, rtol=1e-2), (
        f"max abs err {(o.float() - ref.float()).abs().max().item():.3e}"
    )


def test_flash_decoding_is_causal_correct() -> None:
    # a query attends to all cached keys equally (decode is non-causal over prefix)
    torch.manual_seed(1)
    q = torch.randn(1, 1, 1, 32, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, 512, 32, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 1, 512, 32, device="cuda", dtype=torch.float16)
    o = flash_decoding.flash_decoding(q, k, v, split_size=128)
    ref = flash_decoding.eager_decode(q, k, v)
    assert torch.allclose(o.float(), ref.float(), atol=1e-2, rtol=1e-2)


def test_kv_cache_append_and_reset() -> None:
    cache = flash_decoding.KVCache(batch=1, heads=2, head_dim=16, max_len=8,
                                   device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 1, 16, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, 1, 16, device="cuda", dtype=torch.float16)
    kc, vc = cache.append(k, v)
    assert cache.pos == 1 and kc.shape == (1, 2, 1, 16)
    assert torch.equal(kc[0, 0, 0], k[0, 0, 0])
    cache.reset()
    assert cache.pos == 0
