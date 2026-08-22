"""GPTQ-style post-training quantization, from scratch.

GPTQ (Frantar et al., 2022) quantizes a trained network layer by layer. For
each linear layer it:

1. estimates the layer's **Hessian** ``H = XᵀX`` from calibration activations
   ``X`` (the second-order Taylor term of the loss w.r.t. the weights);
2. greedily quantizes the weight matrix **column by column**, and after each
   column, **updates the remaining columns to absorb that column's rounding
   error** — using the inverse Hessian so the correction is optimal in the
   least-squares sense (the OBQ update).

The result is a quantized model that stays much closer to the fp32 output than
naive round-to-nearest, because the error of each weight is *moved* onto the
weights that follow it rather than just thrown away.

This module applies GPTQ to PicoLM (a from-scratch GPT) and reports the honest
perplexity-vs-bits curve against naive RTN.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from picolm.model import GPT
else:
    # Keep this module importable before the sibling PicoLM checkout is located.
    GPT = Any


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------
def quantize_col(w: torch.Tensor, bits: int, group_size: int | None = None) -> torch.Tensor:
    """Symmetric round-to-nearest quantization of a column ``[out]``.

    ``group_size=None`` uses a single per-column scale; otherwise output rows are
    quantized in groups of ``group_size`` with a shared scale per group (the
    standard LLM group-quant scheme).
    """
    if bits >= 16:
        return w
    qmax = 2 ** (bits - 1) - 1  # e.g. 127 for int8, 3 for 2-bit
    if group_size is None:
        scale = w.abs().max().clamp_min(1e-12) / qmax
        return torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
    wq = w.clone()
    for g in range(0, w.numel(), group_size):
        sl = slice(g, min(g + group_size, w.numel()))
        scale = w[sl].abs().max().clamp_min(1e-12) / qmax
        wq[sl] = torch.clamp(torch.round(w[sl] / scale), -qmax, qmax) * scale
    return wq


def rtn_layer(w: torch.Tensor, bits: int, group_size: int | None = None) -> torch.Tensor:
    """Naive round-to-nearest over every column — the GPTQ baseline."""
    wq = w.float().clone()
    for j in range(wq.shape[1]):
        wq[:, j] = quantize_col(wq[:, j], bits, group_size)
    return wq.to(w.dtype)


def gptq_layer(
    w: torch.Tensor, x: torch.Tensor, bits: int,
    damp: float = 0.01, act_order: bool = True, group_size: int | None = None,
) -> torch.Tensor:
    """GPTQ-quantize one linear layer ``w`` (``[out, in]``) given activations ``x``.

    ``x`` has shape ``[n, in]`` (calibration samples flattened over batch and
    sequence). ``act_order`` reorders input columns by activation energy (the
    activation-ordering trick from the GPTQ paper) so the highest-energy columns
    are quantized first and their error is absorbed by the rest. ``group_size``
    (optional) quantizes output rows in groups with a shared scale.
    """
    if bits >= 16:
        return w.clone()
    out_dim, in_dim = w.shape
    n = x.shape[0]

    # Hessian H = X^T X / n, with damping for numerical stability.
    xf = x.float()
    H = (xf.t() @ xf) / n
    act_mag = torch.diag(H).clone()  # activation energy per input column (pre-damping)
    diag_mean = act_mag.mean()
    H = H + damp * diag_mean * torch.eye(in_dim, device=H.device)
    H_inv = torch.linalg.inv(H)  # [in, in]

    wq = w.float().clone()

    order = None
    if act_order:
        order = torch.argsort(act_mag, descending=True)
        wq = wq[:, order]
        H_inv = H_inv[order][:, order]

    for j in range(in_dim):
        col = wq[:, j]                      # [out]
        q = quantize_col(col, bits, group_size)  # [out]
        err = col - q                       # [out] — the rounding error
        ratio = H_inv[j, j + 1:] / H_inv[j, j]  # [in - j - 1]
        wq[:, j + 1:] -= torch.outer(err, ratio)  # absorb the error
        wq[:, j] = q

    if order is not None:
        wq = wq[:, torch.argsort(order)]  # un-permute back to original column order

    return wq.to(w.dtype)


# ---------------------------------------------------------------------------
# Layer-wise application (sequential re-capture = the correct GPTQ loop)
# ---------------------------------------------------------------------------
_LINEAR_NAMES = ("c_attn", "c_proj", "c_fc", "c_proj")


def _linear_layers(model: GPT) -> list[tuple[str, nn.Linear]]:
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and name.startswith("transformer.h")
    ]


def _capture_inputs(
    model: GPT,
    module: nn.Linear,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    num_batches: int,
) -> torch.Tensor:
    """Run calibration batches through the (partially quantized) model and
    collect this layer's input activations."""
    from picolm.training import get_batch  # picolm only needed for the driver

    inputs: list[torch.Tensor] = []

    def hook(_m, args):
        inputs.append(args[0].detach())

    handle = module.register_forward_pre_hook(hook)
    model.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            x, _ = get_batch(data, block_size, batch_size, device)
            model(x)
    handle.remove()
    return torch.cat(inputs, dim=0).reshape(-1, inputs[0].shape[-1])


def quantize_model(
    model: GPT,
    data: torch.Tensor,
    bits: int,
    *,
    method: str = "gptq",
    block_size: int,
    batch_size: int,
    device: torch.device,
    num_batches: int,
) -> GPT:
    """Quantize every transformer linear layer to ``bits`` with ``method``."""
    layers = _linear_layers(model)
    for i, (name, mod) in enumerate(layers):
        if method == "gptq":
            x = _capture_inputs(model, mod, data, block_size, batch_size, device, num_batches)
            wq = gptq_layer(mod.weight.data, x, bits)
        else:  # rtn — no calibration needed, round each weight directly
            wq = rtn_layer(mod.weight.data, bits)
        mod.weight.data = wq
        print(f"    [{i + 1}/{len(layers)}] quantized {name:<28} -> {bits}-bit")
    return model


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="GPTQ vs RTN quantization of PicoLM")
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--text", default="data/input.txt")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--calib-batches", type=int, default=32)
    ap.add_argument("--eval-batches", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    # A sibling PicoLM checkout uses a ``src/`` package layout.  Make the
    # documented ``--ckpt <picolm>/out/ckpt.pt`` command work without requiring
    # users to editable-install PicoLM into the kernel environment first.
    ckpt_path = Path(args.ckpt).resolve()
    picolm_src = ckpt_path.parent.parent / "src"
    if (picolm_src / "picolm").is_dir():
        sys.path.insert(0, str(picolm_src))

    try:
        from picolm.cli import _load_tokenizer
        from picolm.eval import model_perplexity
        from picolm.model import GPT
    except ModuleNotFoundError as exc:
        if exc.name == "picolm":
            raise SystemExit(
                "PicoLM is not importable. Point --ckpt at a PicoLM checkout "
                "(<repo>/out/ckpt.pt), or install it with `pip install -e <repo>`."
            ) from exc
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"device: {device_name} | checkpoint: {args.ckpt}")

    tok = _load_tokenizer(Path(args.ckpt).parent)
    ids = torch.tensor(tok.encode(Path(args.text).read_text(encoding="utf-8")), dtype=torch.long)
    n = int((1.0 - args.val_frac) * len(ids))
    train_data, val_data = ids[:n], ids[n:]

    cfg = GPT.load(args.ckpt).config
    block_size = cfg.block_size

    def _ppl(model: GPT) -> float:
        _, ppl = model_perplexity(
            model, val_data, block_size, args.batch_size, device, num_batches=args.eval_batches
        )
        return ppl

    print("\n=== baseline (unquantized) ===")
    base = GPT.load(args.ckpt).to(device).eval()
    base_ppl = _ppl(base)
    print(f"    fp model perplexity: {base_ppl:.3f}")

    print("\n=== quantizing (RTN vs GPTQ) ===")
    bits_list = [8, 6, 4, 3, 2]
    rows = []
    for bits in bits_list:
        for method in ("rtn", "gptq"):
            m = GPT.load(args.ckpt).to(device).eval()
            print(f"\n  --- {method.upper()} {bits}-bit ---")
            quantize_model(
                m, train_data, bits, method=method, block_size=block_size,
                batch_size=args.batch_size, device=device,
                num_batches=args.calib_batches,
            )
            ppl = _ppl(m)
            print(f"    {method:>5} {bits}-bit perplexity: {ppl:.3f}")
            rows.append((method, bits, ppl))

    print("\n=== summary (perplexity, lower is better) ===")
    print(f"{'method':>6} {'bits':>4} {'perplexity':>11} {'delta':>9}")
    print("-" * 36)
    print(f"{'fp':>6} {'-':>4} {base_ppl:>11.3f} {'-':>9}")
    for method, bits, ppl in rows:
        print(f"{method:>6} {bits:>4} {ppl:>11.3f} {ppl - base_ppl:>+9.3f}")


if __name__ == "__main__":
    sys.exit(main())
