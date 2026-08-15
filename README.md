# ⚡ PicoKernels — GPU kernels from scratch

**From-scratch GPU kernels in Triton — a GEMM that matches cuBLAS, and a FlashAttention that beats torch's fused attention.**

No `cublas`, no vendor calls. Just the tiling, shared-memory blocking, tensor-core
`tl.dot`, L2-cache swizzling, and online-softmax — written from first principles
and benchmarked rigorously.

## Results

Measured on an NVIDIA RTX 5070 Laptop GPU (Blackwell, sm_120), `torch 2.11.0+cu128`,
Triton 3.7.1.

### GEMM

Numbers are **Triton ÷ cuBLAS** (ratio of GFLOPS), so **> 1.0 means the
from-scratch kernel is faster than NVIDIA's library**:

| dtype | 512² | 1024² | 2048² | 4096² |
|-------|------|-------|-------|-------|
| fp16  | 1.08× | 1.03× | 0.98× | 1.14× |
| bf16  | 1.02× | 0.98× | 1.08× | 0.96× |

Peak throughput ~47–48 TFLOPS fp16 at 4096². Correctness is verified against
`torch.matmul` (max abs err ≈ 3e-4 for fp16, ≈ 2e-4 for bf16 — pure fp32-accumulate
rounding noise).

### FlashAttention (causal forward)

Speedup vs eager attention and vs `torch.nn.functional.scaled_dot_product_attention`
(cuDNN's fused FlashAttention). Batch 4, 8 heads, head dim 64, fp16:

| N (seq len) | vs eager | vs torch SDPA |
|-------------|----------|---------------|
| 256   | 9.2×  | 1.69× |
| 512   | 24.7× | 1.80× |
| 1024  | 34.0× | 1.87× |
| 2048  | 37.7× | 1.88× |
| 4096  | 39.3× | 1.79× |

The from-scratch kernel is **~1.8× faster than NVIDIA's own fused attention** —
the single-pass online-softmax avoids materializing the `N × N` score matrix,
which is what makes the O(N²)→O(N) memory win and the speed win concrete.
(Comparison includes torch SDPA's per-call dispatch overhead.)

## Why it's fast (GEMM)

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
gemm.py                  # tiled GEMM + autotune + benchmark
attention.py             # FlashAttention (causal) + benchmark
tests/test_gemm.py       # GEMM correctness (GPU-gated)
tests/test_attention.py  # attention correctness + causality (GPU-gated)
```

## Roadmap

Next: GPTQ-style quantization from scratch, then a from-scratch diffusion model —
each benchmarked with the same rigor.
