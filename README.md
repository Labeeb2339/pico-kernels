# ⚡ PicoKernels — GPU kernels from scratch

**From-scratch GPU kernels in Triton — a GEMM that matches cuBLAS, and a FlashAttention that beats torch's fused attention.**

No `cublas`, no vendor calls. Just the tiling, shared-memory blocking, tensor-core
`tl.dot`, L2-cache swizzling, and online-softmax — written from first principles
and benchmarked rigorously.

## Results

Measured on an NVIDIA RTX 5070 Laptop GPU (Blackwell, sm_120), `torch 2.11.0+cu128`,
Triton 3.7.1.

Every number below is reproduced by one command:

```bash
python bench.py          # run all ten benchmarks sequentially
python bench.py --log    # also write the raw output to bench_output.txt
```

For an engineer-reviewable causal-attention run, use the provenance collector.
It runs correctness before performance and records the Git/source fingerprint,
runtime and GPU environment, exact commands, path-scrubbed logs, exit codes, and log
hashes. The source-to-invariant map and claim boundaries are documented in
[`docs/attention_evidence.md`](docs/attention_evidence.md).

```powershell
$run = Get-Date -Format 'yyyyMMdd-HHmmss'
.\.venv\Scripts\python.exe scripts\collect_attention_evidence.py `
  --output-dir "evidence\runs\$run-attention"
```

The current checked-in receipt was captured from clean source commit
[`b8d6936`](https://github.com/Labeeb2339/pico-kernels/commit/b8d69367116c06835ffe6ebf850b7c59cf05160c)
on 2026-08-22. All nine attention correctness/API cases passed. Under the
recorded fixed protocol, the Triton kernel measured **1.79x faster than PyTorch
SDPA at N=4096** (1.927 ms versus 3.448 ms) and **1.84x at N=32768** (110.452 ms
versus 203.525 ms). These are machine-specific medians, not portable speed
guarantees. See the [receipt](evidence/runs/20260822-2332-attention/receipt.json)
and [captured benchmark log](evidence/runs/20260822-2332-attention/02-benchmark.stdout.txt).

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

### Raw CUDA (below Triton)

`raw_gemm.py` drops one layer lower: a hand-written CUDA C GEMM (16×16
shared-memory tile, one output element per thread, no tensor cores) compiled at
runtime by the NVRTC that ships inside the torch wheel and launched through the
CUDA driver API — no Triton, no cuBLAS, no toolkit install.

| kernel | fp32 TFLOPS (1024³) |
|--------|---------------------|
| cuBLAS (torch matmul) | 12.2 |
| raw CUDA C (this)     | 1.9  |

**6.3× slower than cuBLAS** — and that gap *is* the point. Tensor cores and fine
register tiling are exactly what cuBLAS/Triton add on top of a raw baseline; this
kernel makes those layers concrete. Correctness is verified against `torch.matmul`
(max abs err ≈ 2e-4).

### fp8 GEMM (Blackwell)

Blackwell (sm_120) adds native fp8 (e4m3) tensor cores at ~2× the fp16 rate.
`gemm_fp8.py` is a *scaled*-fp8 matmul: inputs are cast to e4m3 with per-tensor
scale factors, the tensor-core matmul runs in fp8 with an fp32 accumulator, and
the output is rescaled once in the epilogue.

| size  | fp16 cuBLAS | fp8 triton | speedup |
|-------|-------------|------------|---------|
| 512²  | 18.8 T      | 21.0 T     | 1.11×   |
| 1024² | 35.2 T      | 56.6 T     | 1.61×   |
| 2048² | 46.1 T      | 76.7 T     | 1.67×   |
| 4096² | 39.4 T      | 84.2 T     | **2.14×** |

**~2.1× faster than cuBLAS fp16 at 4096²** — the fp8 tensor-core rate advantage,
for real. Accuracy is the honest cost: **3.74% relative Frobenius error vs fp32**
(e4m3 has a 3-bit mantissa). The cast/scale is an O(n²) memory pass done once
(offline for weights); the O(n³) matmul is what's timed.

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

### FlashAttention backward

`attention_bwd.py` computes `dQ, dK, dV` from `dO` — also fused, and also without
materializing `N × N`. Softmax stats are recomputed in a first pass, then
`dV = P^T dO`, `dP = dO V^T`, `dS = P ⊙ (dP - D)` with `D = rowsum(dO·O)`, and
`dQ/dK` accumulate across key tiles (with atomics for the cross-tile reductions).
Gradients match PyTorch autograd of an eager attention to **~1e-3 max abs err
(fp16) / ~6e-3 (bf16)**.

Forward+backward speedup vs eager (which materializes `N × N`):

| N    | vs eager (fwd+bwd) |
|------|--------------------|
| 256  | 3.75×              |
| 512  | 2.69×              |
| 1024 | 5.69×              |
| 2048 | 6.93×              |

### FlashDecoding + KV cache (decode)

`flash_decoding.py` adds the two pieces that make *generation* fast: a **KV
cache** (stores past keys/values so each token attends only to the cached
prefix) and a **FlashDecoding-style split-KV kernel** — during decode `M=1`, so
the KV sequence is split across blocks and the per-split `(max, sum, acc)` are
reduced with the same online-softmax rescaling. Speedup vs eager single-token
decode:

| N    | vs eager decode |
|------|-----------------|
| 1024 | 1.4×            |
| 2048 | 5.8×            |
| 4096 | 2.4×            |
| 8192 | 5.7×            |

### Grouped-Query Attention (GQA)

`attention_gqa.py` extends the causal FlashAttention to GQA (`H_q > H_kv`), the
head structure used by Llama/Mistral/Gemma — each KV head is shared by
`H_q / H_kv` query heads, shrinking the KV cache and KV bandwidth. Correctness
is verified against an eager GQA reference; the fused kernel is **~34–38×
faster** (the eager reference materializes the `N × N` matrix via KV repeat, the
kernel never does).

### GPTQ quantization (applied to PicoLM)

Perplexity of PicoLM (a 10.6M-param from-scratch GPT, fp perplexity 4.62) after
quantizing all 24 transformer linears with naive round-to-nearest (RTN) vs GPTQ's
inverse-Hessian error compensation:

| bits | RTN ppl | GPTQ ppl |
|------|---------|----------|
| 8    | 4.59    | 4.59     |
| 6    | 4.58    | 4.61     |
| **4**| 4.71    | **4.63** |
| **3**| 5.32    | **4.99** |
| 2    | 33.19   | **20.75**|

At **4-bit, GPTQ is essentially lossless** (+0.013 ppl ≈ 0.3%) while RTN degrades
~7× more (+0.088). The inverse-Hessian update moves each weight's rounding error
onto the weights that follow it, so the layer output stays near-identical — 4-bit
is 4× smaller than fp16, 8× smaller than fp32, at almost no quality cost.

`gptq_layer` also supports **activation-ordering** (`act_order`, the column-reorder
trick from the GPTQ paper — highest-energy columns quantized first) and
**per-group output quantization** (`group_size`, shared scale per output group).
Act-ordering is permutation-consistent (unit-tested); its benefit over the OBQ
update alone is modest in synthetic tests but is the standard quality lever for
3-bit/2-bit.

Run it against a PicoLM checkpoint: `python quantize.py --ckpt <picolm>/out/ckpt.pt`.

**Why "GPTQ vs RTN", not "GPTQ vs AutoGPTQ / llama.cpp".** AutoGPTQ implements
the *same* GPTQ algorithm (Frantar et al., 2022) — a head-to-head would just
reproduce these numbers, which tells you nothing new. llama.cpp quantizes with a
different family entirely (GGML block-wise quants like `Q4_K_M`), so comparing
perplexity is apples-to-oranges across methods rather than a validation of this
one. The honest question "does GPTQ actually help?" is answered by the RTN
baseline, which the table above isolates directly.

### INT4 inference (the speedup behind 4-bit GPTQ)

`gemm_int4.py` packs 4-bit weights 2-per-byte and decompresses them in-kernel.
The win is **bandwidth, not FLOPs**: a quantized linear layer reads 4× fewer
weight bytes, which pays off exactly in the memory-bound regime (small batch =
LLM decode). Measured on this GPU:

| shape (M,N,K)      | fp16   | int4   | speedup | regime        |
|--------------------|--------|--------|---------|---------------|
| (16, 4096, 4096)   | 5.3 T  | 6.3 T  | 1.19×   | memory-bound  |
| (16, 8192, 8192)   | 5.3 T  | 7.5 T  | 1.40×   | memory-bound  |
| (1024, 4096, 4096) | 45.9 T | 36.6 T | 0.80×   | mixed         |
| (4096, 4096, 4096) | 44.7 T | 30.0 T | 0.67×   | compute-bound |

**~1.2–1.4× faster for decode-shaped matmuls, not faster (and slightly slower)
for compute-bound ones** — the honest two-regime picture of 4-bit inference.
Decompression is exact (0% error vs the fp16-quantized reference).

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
gemm_fp8.py              # fp8 (e4m3) scaled GEMM + benchmark (Blackwell)
gemm_int4.py             # packed INT4 GEMM + benchmark (memory-bound win)
gemm_splitk.py           # split-K GEMM (documented failure case)
attention.py             # FlashAttention (causal) + benchmark
attention_bwd.py         # FlashAttention backward (fused) + benchmark
flash_decoding.py        # KV cache + FlashDecoding (decode)
quantize.py              # GPTQ vs RTN quantization (applied to PicoLM)
tests/test_gemm.py       # GEMM correctness (GPU-gated)
tests/test_gemm_fp8.py   # fp8 GEMM correctness (GPU-gated)
tests/test_gemm_int4.py  # INT4 GEMM correctness (GPU-gated)
tests/test_gemm_splitk.py  # split-K GEMM correctness (GPU-gated)
tests/test_attention.py  # attention correctness + causality (GPU-gated)
tests/test_attention_bwd.py  # backward correctness vs autograd (GPU-gated)
tests/test_flash_decoding.py  # flash-decoding + KV cache correctness (GPU-gated)
```

## Roadmap

The from-scratch diffusion model (DDPM + DDIM U-Net) lives in its own repo:
[`pico-diffusion`](https://github.com/Labeeb2339/pico-diffusion) — same rigor,
different modality. A from-scratch GGUF inference engine (GGML dequant + LLaMA
forward + BPE) lives in [`pico-engine`](https://github.com/Labeeb2339/pico-engine),
which serves a real Qwen2.5 model end to end.
