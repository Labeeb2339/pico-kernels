# ⚡ PicoKernels — GPU kernels from scratch

**A from-scratch matrix-multiplication (GEMM) kernel in Triton that holds parity with — and often beats — NVIDIA's cuBLAS.**

No `cublas`, no vendor calls. Just the tiling, shared-memory blocking, tensor-core
`tl.dot`, and L2-cache swizzling that make GEMM fast, written from first
principles and autotuned.

## Results

Measured on an NVIDIA RTX 5070 Laptop GPU (Blackwell, sm_120), `torch 2.11.0+cu128`,
Triton 3.7.1. Numbers are **Triton ÷ cuBLAS** (ratio of GFLOPS), so **> 1.0 means the
from-scratch kernel is faster than NVIDIA's library**:

| dtype | 512² | 1024² | 2048² | 4096² |
|-------|------|-------|-------|-------|
| fp16  | 1.08× | 1.03× | 0.98× | 1.14× |
| bf16  | 1.02× | 0.98× | 1.08× | 0.96× |

Peak throughput ~47–48 TFLOPS fp16 at 4096². Correctness is verified against
`torch.matmul` (max abs err ≈ 3e-4 for fp16, ≈ 2e-4 for bf16 — pure fp32-accumulate
rounding noise).

## Why it's fast

1. **Tiling** — the `M × N` output is split into `BLOCK_M × BLOCK_N` tiles and the
   `K` reduction is chunked into `BLOCK_K` steps, so each tile stays in registers
   and shared memory instead of thrashing global memory.
2. **fp32 accumulation on tensor cores** — `tl.dot` maps to the tensor-core
   matrix-multiply-accumulate instruction, accumulating in fp32 while inputs stay
   fp16/bf16 (the whole point of tensor cores).
3. **L2 swizzle** — programs launch in `GROUP_M` groups so tiles sharing the same
   A-rows run together, maximizing reuse of B from L2 cache.
4. **Autotuning** — `@triton.autotune` sweeps 8 tile shapes × warp counts ×
   pipeline stages and caches the fastest per `(M, N, K)`. The optimal shape is
   *not* universal: fp16 favours `64×128` tiles, bf16 favours `64×64` — autotune
   finds whichever wins for the actual shape.

> fp32 is intentionally excluded: `tl.dot` routes fp32 through tensor cores with
> TF32 (10-bit mantissa) — the same reduced-precision tradeoff cuBLAS makes — so
> fp16/bf16 are the honest tensor-core benchmark.

## Quickstart

```bash
# 1. venv + PyTorch with CUDA 12.8
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. Triton (Windows uses the community build; Linux/Mac just `pip install triton`)
pip install triton-windows                                # Windows
# pip install triton                                      # Linux / WSL2

# 3. Run the benchmark
python gemm.py
```

## Files

```
gemm.py              # the kernel + autotune + benchmark
tests/test_gemm.py   # correctness checks (GPU-gated, skips on CPU)
```

## Roadmap

This is the first rung of a from-scratch systems stack. Next: a FlashAttention-style
fused attention kernel (the same GEMM structure plus an online-softmax), then
GPTQ-style quantization — each benchmarked with the same rigor.
