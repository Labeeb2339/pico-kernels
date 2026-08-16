#!/usr/bin/env python3
"""Run every kernel benchmark in one command and print a unified summary.

Each kernel module ships a self-contained correctness + benchmark block under
``if __name__ == "__main__"``. This runner executes them all sequentially, so
``python bench.py`` reproduces the whole "Results" section of the README on a
single GPU.

The GPTQ kernel applies quantization to a trained PicoLM checkpoint, so it
needs a sibling PicoLM checkout. Pass ``--picolm-dir`` to point at it (default
``../PicoLM``); the runner skips GPTQ with a note if no checkpoint is found.

Usage:
    python bench.py                 # run everything, print the results
    python bench.py --log           # also write full raw output to bench_output.txt
    python bench.py --skip gptq     # skip a kernel by name substring
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (name, module)
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


def run_kernel(module: str, extra_args: tuple[str, ...] = ()) -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / module), *extra_args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return f"[FAILED exit {result.returncode}]\n{result.stderr[-2000:]}"
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="store_true", help="write full raw output to bench_output.txt")
    parser.add_argument("--skip", action="append", default=[], help="skip a kernel by name substring")
    parser.add_argument("--picolm-dir", default="../PicoLM", help="sibling PicoLM checkout for the GPTQ kernel")
    args = parser.parse_args()

    picolm = (ROOT / args.picolm_dir).resolve()
    gptq_ckpt = picolm / "out" / "ckpt.pt"
    gptq_text = picolm / "data" / "input.txt"

    python = Path(sys.executable)
    print(f"=== pico-kernels benchmark suite ({python.name} {sys.version.split()[0]}) ===\n")

    full_log = []
    for name, module in KERNELS:
        if any(s.lower() in name.lower() for s in args.skip):
            print(f"[skip] {name}")
            continue
        print(f"--- {name} ---")
        extra_args: tuple[str, ...] = ()
        if module == "quantize.py":
            if not gptq_ckpt.exists():
                print(f"[skip] GPTQ needs a PicoLM checkpoint; none at {gptq_ckpt} "
                      f"(pass --picolm-dir or train PicoLM first)\n")
                continue
            extra_args = ("--ckpt", str(gptq_ckpt), "--text", str(gptq_text))
        output = run_kernel(module, extra_args)
        full_log.append(f"===== {name} ({module}) =====\n{output}\n")
        print(output.strip()[-4000:])
        print()

    if args.log:
        Path("bench_output.txt").write_text("".join(full_log), encoding="utf-8")
        print("full raw output written to bench_output.txt")


if __name__ == "__main__":
    main()
