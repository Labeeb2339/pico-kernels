#!/usr/bin/env python3
"""Run every kernel benchmark in one command and print a unified summary.

Each kernel module ships a self-contained correctness + benchmark block under
``if __name__ == "__main__"``. This runner executes them all sequentially and
collects the headline number from each, so ``python bench.py`` reproduces the
whole "Results" section of the README on a single GPU.

Usage:
    python bench.py            # run everything, print a summary
    python bench.py --log      # also write the full raw output to bench_output.txt
    python bench.py --skip gptq # skip a kernel by name
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (name, module, headline description)
KERNELS = (
    ("GEMM (fp16/bf16 vs cuBLAS)", "gemm.py"),
    ("fp8 GEMM (Blackwell)", "gemm_fp8.py"),
    ("FlashAttention forward", "attention.py"),
    ("FlashAttention backward", "attention_bwd.py"),
    ("Grouped-Query Attention", "attention_gqa.py"),
    ("FlashDecoding + KV cache", "flash_decoding.py"),
    ("INT4 GEMM", "gemm_int4.py"),
    ("Split-K GEMM (failure case)", "gemm_splitk.py"),
    ("GPTQ vs RTN perplexity", "quantize.py"),
)


def run_kernel(module: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / module)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return f"[FAILED exit {result.returncode}]\n{result.stderr[-2000:]}"
    return result.stdout


def headline(name: str, output: str) -> str:
    """Pull the single most interesting number out of a kernel's output."""
    patterns = (
        # gemm.py: "fp16  ... 1.14x"
        (r"(fp16\s+\S+\s+\S+\s+\S+\s+[\d.]+x)", "fp16 GEMM vs cuBLAS"),
        # gemm_fp8.py: "4096²  39.4 T  84.2 T  2.14×"
        (r"(4096²\s+[\d.]+\s*T\s+[\d.]+\s*T\s+([\d.]+)×)", "fp8 GEMM speedup @4096²"),
        # attention.py: "4096  39.3x ..." (vs eager)
        (r"(4096\s+[\d.]+x\s+[\d.]+x)", "FlashAttention vs eager @4096"),
        # attention_bwd.py: "2048  6.93x"
        (r"(2048\s+[\d.]+x)", "FA backward @2048"),
        # attention_gqa.py: max speedup
        (r"(gqa[\s\S]{0,120}?[\d.]+x)", "GQA"),
        # flash_decoding.py
        (r"(2048\s+[\d.]+x)", "FlashDecoding @2048"),
        # gemm_int4.py
        (r"(decode[\s\S]{0,80}?[\d.]+x)", "INT4 decode"),
        # quantize.py: "4  4.71  4.63"
        (r"(\s*4\s+[\d.]+\s+[\d.]+)", "GPTQ 4-bit ppl"),
    )
    for pat, _label in patterns:
        m = re.search(pat, output)
        if m:
            return m.group(1).strip()
    return "(see output)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="store_true", help="write full raw output to bench_output.txt")
    parser.add_argument("--skip", action="append", default=[], help="skip a kernel by name substring")
    args = parser.parse_args()

    python = Path(sys.executable)
    print(f"=== pico-kernels benchmark suite ({python.name} {sys.version.split()[0]}) ===\n")

    full_log = []
    for name, module in KERNELS:
        if any(s.lower() in name.lower() for s in args.skip):
            print(f"[skip] {name}")
            continue
        print(f"--- {name} ---")
        output = run_kernel(module)
        full_log.append(f"===== {name} ({module}) =====\n{output}\n")
        print(output.strip()[-4000:])
        print()

    if args.log:
        Path("bench_output.txt").write_text("".join(full_log), encoding="utf-8")
        print("full raw output written to bench_output.txt")


if __name__ == "__main__":
    main()
