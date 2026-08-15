"""Correctness tests for the GPTQ primitives (act-ordering, group quant)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # pico-kernels
sys.path.insert(0, r"C:/Users/Labeeb/PicoLM/src")                   # picolm (quantize imports it)

import quantize  # noqa: E402
from quantize import gptq_layer, quantize_col, rtn_layer  # noqa: E402


def test_gptq_act_order_is_permutation_consistent() -> None:
    torch.manual_seed(0)
    W = torch.randn(64, 32)
    X = torch.randn(16, 32)
    X[:, :8] *= 4.0  # skewed activation energy

    w_ao = gptq_layer(W, X, 4, act_order=True)

    # manual equivalent: permute by activation energy, quantize (no act-order),
    # then un-permute — must be identical to the act_order=True path
    xf = X.float()
    H = (xf.t() @ xf) / X.shape[0]
    act_mag = torch.diag(H)
    order = torch.argsort(act_mag, descending=True)
    w_manual = gptq_layer(W[:, order], X[:, order], 4, act_order=False)
    w_manual = w_manual[:, torch.argsort(order)]

    assert torch.allclose(w_ao, w_manual, atol=1e-5), "act-ordering is not permutation-consistent"


def test_quantize_col_group_size_matches_manual() -> None:
    torch.manual_seed(1)
    w = torch.randn(64)
    q = quantize_col(w, 4, group_size=16)
    for g in range(0, 64, 16):
        scale = w[g:g + 16].abs().max() / 7.0
        expected = torch.clamp(torch.round(w[g:g + 16] / scale), -7, 7) * scale
        assert torch.allclose(q[g:g + 16], expected, atol=1e-5)


def test_gptq_beats_rtn_at_4bit() -> None:
    torch.manual_seed(2)
    W = torch.randn(128, 256) * 0.1
    X = torch.randn(64, 256)

    def mse(wq):
        return ((X @ wq.T - X @ W.T) ** 2).mean().item()

    assert mse(gptq_layer(W, X, 4)) < mse(rtn_layer(W, 4))
