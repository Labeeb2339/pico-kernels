"""Correctness test for the raw CUDA C GEMM (below Triton).

GPU-gated. Also skipped when ``cuda-python`` is absent, since the kernel is
compiled + launched through the bundled NVRTC + driver API.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import raw_gemm  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_raw_gemm_matches_cublas():
    pytest.importorskip("cuda", reason="cuda-python required for the raw kernel")
    run = raw_gemm.build_sgemm()
    M = N = K = 256
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    run(A, B, C, M, N, K)
    ref = A @ B
    assert (C - ref).abs().max().item() < 1e-3
