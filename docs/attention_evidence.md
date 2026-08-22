# FlashAttention evidence map

This is the review path for the causal forward kernel. It links each claim to
the implementation, the invariant that can falsify it, and the measurement that
is allowed to support it. Performance numbers are measurements, not guarantees.

## Scope

`triton_attention(q, k, v)` implements:

```text
softmax((Q @ K^T) / sqrt(D) + causal_mask) @ V
```

The public wrapper accepts contiguous CUDA tensors with identical `(B, H, N, D)`
shapes and fp16 or bf16 dtype. `D` is a power of two at least 16. `N` is
divisible by both tile sizes. The implementation has no dropout, additive
attention bias, padding mask, cross-attention, ragged sequence, or backward pass.
Those are deliberate boundaries, not implied features.

## Source → invariant → evidence

| Source | Mechanism | Falsifiable invariant | Evidence |
|---|---|---|---|
| `attention.py::attn_fwd` | Tiles Q/K/V and keeps only a running row max, row sum, and output accumulator | No global `N x N` score or probability tensor is allocated by this kernel | Static source inspection; use a profiler separately if allocation evidence is required |
| `attention.py::attn_fwd` | Applies `offs_m >= start_n + offs_n` before exponentiation | Mutating a future key/value cannot change an earlier query output | `tests/test_attention.py::test_attention_causality` |
| `attention.py::attn_fwd` | Updates online-softmax state with `m_i`, `l_i`, and `alpha` | Output matches the mathematically direct fp32 eager implementation | `test_attention_matches_naive`, fp16 and bf16, `N ∈ {128,256,512}`, `atol=rtol=5e-2` |
| `attention.py::triton_attention` | Checks device, dtype, shape, layout, tile divisibility, and head dimension before launch | Unsupported inputs fail before JIT launch with a specific exception | Input-contract tests in `tests/test_attention.py` |
| `attention.py::bench` | Times eager, PyTorch SDPA, and this Triton kernel on identical tensors | Reported ratio equals baseline median latency divided by Triton median latency for that row | Raw `02-benchmark.stdout.txt` plus the receipt's source/environment hashes |

## Correctness guarantees—and no more

The checked cases guarantee all of the following when the tests pass:

- causal semantics with scale `1 / sqrt(D)`;
- parity with both a direct fp32 eager reference and PyTorch SDPA within
  `atol=5e-2, rtol=5e-2` for the parameterized dtype/shape matrix;
- output shape, dtype, and device preservation;
- rejection of noncontiguous, unsupported-dtype, and tile-incompatible inputs.

They do **not** establish correctness for untested head dimensions, sequence
lengths, GPU architectures, Triton versions, gradients, or noncausal features.

## Benchmark protocol

The benchmark uses seed 0 and fixed `(B,H,D)=(4,8,64)` fp16 inputs. Sequence
lengths are 256, 512, 1024, 2048, 4096, and 32768. Each provider gets 25 ms of
warmup and 100 ms of timed repetitions through `triton.testing.do_bench`; the
reported statistic is median latency. Provider order is eager, PyTorch SDPA,
then Triton. Eager is intentionally omitted at 32768 because its explicit score
matrix is not viable on this machine.

This fixed order makes the run easy to reproduce, but it does not eliminate
thermal or clock-order effects. Record AC power/performance mode, close unrelated
GPU workloads, and rerun the full receipt when source, driver, GPU, PyTorch, or
Triton changes. Do not combine rows from different receipts into one table.

## Produce a reviewable receipt

From the repository root in PowerShell:

```powershell
$run = Get-Date -Format 'yyyyMMdd-HHmmss'
.\.venv\Scripts\python.exe scripts\collect_attention_evidence.py `
  --output-dir "evidence\runs\$run-attention"
```

The collector runs the attention test file first and stops if it fails. Only
then does it benchmark. It writes:

- `receipt.json`: source tree, Git HEAD/diff hash, environment, protocol, commands,
  exit codes, and log hashes;
- `README.md`: human-readable receipt and interpretation boundary;
- `01-correctness.stdout.txt` / `.stderr.txt`;
- `02-benchmark.stdout.txt` / `.stderr.txt`.

Use `--metadata-only` for a safe provenance smoke test that does not run CUDA
kernels. A release benchmark receipt should be generated from the exact commit
being presented. If `dirty` is `true`, the receipt still identifies the snapshot
through its complete source-tree hash and HEAD diff hash, but it should be called
a worktree result rather than a commit result.
