# FlashAttention from first principles

This is the write-up for `attention.py`: a causal FlashAttention kernel in
Triton that beats PyTorch's fused SDPA by ~1.8× at these shapes, with zero
pretrained weights and no vendor calls. It exists because the most expensive
question in a transformer is *"can I afford to store the attention matrix?"*

---

## 1. The problem: attention is O(N²) *in memory*

Scaled dot-product attention is

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
```

where `Q, K, V` are `(B, H, N, D)` — batch, heads, sequence length, head dim.

The naive implementation computes `S = Q K^T / sqrt(d)`, an **`N × N` matrix**,
then softmaxes it, then multiplies by `V`. The `N × N` matrix is the killer:

| N (tokens) | entries in S (one head) | fp16 bytes (one head) |
|-----------:|------------------------:|----------------------:|
| 512        | 262k                    | 0.5 MB                |
| 2048       | 4.2M                    | 8.4 MB                |
| 4096       | 16.8M                   | 33.5 MB               |
| 32768      | 1.07 **billion**        | **2.1 GB**            |

Multiply by heads and batch, and a 32k-context model needs terabytes of
intermediate traffic just to *hold* `S` while reading it twice (once to
softmax, once to multiply `V`). That's the wall: attention is **compute-cheap
and memory-bound**, because every element of `S` is written to HBM and read
back, while the actual FLOPs are tiny by comparison.

## 2. The insight: fuse it, tile it, never store S

FlashAttention (Dao et al. 2023, 2024) observes that you don't need the whole
`S` at once. The output row `i` depends only on:

- the `i`-th row of `Q` (a `1 × D` vector), and
- *all* of `K` and `V` (because softmax is a global reduction over the key axis).

So split `Q` into **query tiles** (`BLOCK_M` rows), split `K`/`V` into **key
tiles** (`BLOCK_N` rows), and stream the key tiles through SRAM, accumulating
the result incrementally. The `N × N` matrix is never materialized anywhere:
each tile is computed in registers/SRAM and immediately consumed.

Two things make this work:

1. **The GEMM structure** — each tile of `S` is `tl.dot(q_tile, k_tileᵀ)`, the
   same tiled matmul from `gemm.py`. No new primitive needed.
2. **Online softmax** — softmax is normally a two-pass op (find the max, then
   exp-and-sum), but a running max/sum rescaling makes it a single streaming
   pass. This is the only non-obvious part, so it gets its own section.

## 3. Online softmax — the math

Naive softmax over a row `x₁…xₙ`:

```
m  = max(x)                      # pass 1: stability
l  = Σ exp(xᵢ - m)               # pass 2: denominator
pᵢ = exp(xᵢ - m) / l
```

Two passes over the data. In a streaming kernel you see one tile at a time, so
you can't run pass 1 first. Instead, maintain a **running max** `m` and
**running sum** `l`, and *rescale* the accumulator every time `m` increases:

```
m_new   = max(m_old, max(x_tile))
α       = exp(m_old - m_new)          # rescale factor for old terms
l       = l_old * α + Σ exp(x_tile - m_new)
acc     = acc * α + P_tile @ V_tile   # P_tile = exp(x_tile - m_new)
```

Why this is *exact*, not approximate: every previously-accumulated `exp(x - m_old)`
is multiplied by `α = exp(m_old - m_new) = exp(x - m_old) · exp(m_old - m_new) =
exp(x - m_new)`, which is exactly what it *should* have been under the new max.
Subtracting the running max before every `exp` is what keeps the values from
overflowing to `inf` — the same stability trick as the naive version, done
incrementally. When the loop ends, divide by `l` once.

In `attention.py` this is lines 72–84:

```python
m_ij   = tl.maximum(m_i, tl.max(qk, axis=1))   # running max
p      = tl.exp(qk - m_ij[:, None])            # stable exp under new max
l_ij   = tl.sum(p, axis=1)                     # this tile's contribution
alpha  = tl.exp(m_i - m_ij)                    # rescale old accumulator
l_i    = l_i * alpha + l_ij
acc    = acc * alpha[:, None]                  # rescale old output
acc    = tl.dot(p.to(v.dtype), v, acc)         # accumulate P_tile @ V_tile
m_i    = m_ij
```

## 4. Causal masking, for free

For a decoder, query `i` may only attend to keys `j ≤ i`. The kernel's
`m_tile`-th query tile only iterates keys up to `hi = (m_tile + 1) * BLOCK_M`,
and within the tile it masks future positions:

```python
mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
qk   = tl.where(mask, qk, float("-inf"))       # -inf -> exp -> 0
```

The masked positions contribute `exp(-inf) = 0` to both `l` and `acc`, so the
causal constraint is applied *inside* the same fused pass — no separate masking
step, no wasted compute beyond the half-triangle the loop already skips.

## 5. Correctness (how it was checked)

Nothing here is trusted on vibes. `check_correctness` compares the kernel
against **two independent references**:

- the naive fp32 attention (the mathematically-correct answer), and
- PyTorch's `scaled_dot_product_attention` (NVIDIA's fused kernel).

Result on an RTX 5070 laptop GPU, `B=4, H=8, N=512, D=64`:

| dtype | vs fp32 naive (max abs err) | vs torch SDPA (max abs err) |
|-------|----------------------------:|----------------------------:|
| fp16  | ~1e-3                       | ~1e-3                       |
| bf16  | ~1e-3                       | ~1e-3                       |

The residual is fp16/bf16 *precision*, not the algorithm — online softmax is
bit-for-bit the same math as two-pass softmax. `tests/test_attention.py`
(14 cases) pins this down, including a causality check that verifies every
`i < j` entry is exactly zero.

## 6. Benchmark

`bench()` measures eager (naive `N × N`), `SDPA` (torch's fused kernel), and
this Triton kernel, all on the same tensors. Speedups below were measured on an
RTX 5070 Laptop (sm_120), `B=4, H=8, D=64`, fp16:

| N    | vs eager | vs SDPA |
|-----:|---------:|--------:|
| 256  | 9.2×     | 1.69×   |
| 512  | 24.7×    | 1.80×   |
| 1024 | 34.0×    | 1.87×   |
| 2048 | 37.7×    | 1.88×   |
| 4096 | 39.3×    | 1.79×   |

*(Absolute wall-clock times are hardware-dependent — run `python attention.py`
for your machine's raw ms.)*

Two numbers to read honestly:

- **vs eager** (9–39×): this is the O(N²)-materialization win. Eager's cost is
  dominated by writing and re-reading the `N × N` score matrix; the fused
  kernel simply never pays that IO.
- **vs SDPA** (~1.8×): SDPA is *also* fused and well-tuned, so this is a real
  kernel-quality win at these shapes — block sizing, occupancy, and avoiding
  SDPA's more general (non-causal-optimized) path.

## 7. What this does *not* prove

In the spirit of the repo (and my profile): the limits, stated plainly.

- **Forward only.** There is no backward pass / flash-decoding here; the kernel
  is a causal forward. Training would need the (much harder) fused backward.
- **fp16/bf16 only.** The kernel casts to half precision; it is not an fp32
  attention and would not be bit-exact in fp32.
- **Fixed `HEAD_DIM = D` as a `tl.constexpr`** — the block sizes assume `N`
  divides `BLOCK_M`/`BLOCK_N`; a fully general kernel re-tiles dynamically.
- **One GPU, one shape family.** The ~1.8× vs SDPA is measured on an RTX 5070
  Laptop (sm_120) at `D=64`; it is not a claim that this beats SDPA everywhere.
- **Not FlashAttention-2's full design.** No warp-level work partitioning or
  split-KV; it's the core tiled + online-softmax structure, which is the part
  that matters for the O(N²)→O(N) memory story.

## Reference

Dao, Fu, Ermon, Rudra, Ré — *FlashAttention: Fast and Memory-Efficient Exact
Attention with IO-Awareness* (NeurIPS 2022); Dao — *FlashAttention-2* (2023).
