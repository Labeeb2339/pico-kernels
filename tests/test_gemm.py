"""Correctness tests for the from-scratch Triton GEMM.

These need a CUDA GPU (they're skipped on CPU, e.g. in CI). They check the
kernel against ``torch.matmul`` across shapes and the tensor-core dtypes.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemm  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def _check(dtype: torch.dtype, m: int, n: int, k: int) -> None:
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    c = gemm.triton_gemm(a, b)
    ref = a.float() @ b.float()
    assert torch.allclose(c, ref, atol=1e-2, rtol=1e-2), (
        f"mismatch at m={m} n={n} k={k} dtype={dtype}: "
        f"max abs err {(c - ref).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)])
def test_gemm_matches_torch(dtype: torch.dtype, shape: tuple[int, int, int]) -> None:
    m, n, k = shape
    _check(dtype, m, n, k)


def test_gemm_output_dtype_is_fp32() -> None:
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    c = gemm.triton_gemm(a, b)
    assert c.dtype == torch.float32
