# PicoKernels attention evidence receipt

Generated: `2026-08-22T15:32:29.461808+00:00`

## Source identity

- Git HEAD: `b8d69367116c06835ffe6ebf850b7c59cf05160c`
- Branch: `main`
- Dirty worktree: `no`
- Diff SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Complete source-tree SHA-256: `9de483d51f893dc85a64dace88bac9e3c1f2494267bf97fcb2ef4554d45e8b50`
- Source stable during run: `yes`

## Environment

- Python: `3.11.15`
- PyTorch: `2.11.0+cu128` (CUDA `12.8`)
- Triton: `3.7.1`
- GPU: `NVIDIA GeForce RTX 5070 Laptop GPU`

## Contract under test

- `ATTN-SEM-001` — Implements causal softmax(QK^T / sqrt(D))V with no dropout, bias, or padding mask.
- `ATTN-NUM-001` — Matches an fp32 eager reference and torch SDPA at atol=5e-2, rtol=5e-2.
- `ATTN-CAUSAL-001` — Changing the final key/value cannot change earlier query outputs.
- `ATTN-API-001` — Output preserves q shape, dtype, and CUDA device; unsupported layouts fail explicitly.
- `ATTN-MEM-001` — The Triton kernel tiles K/V and never stores an N x N score tensor in global memory.

## Sequential commands

### 01-correctness

- Command: `.venv/Scripts/python.exe -m pytest tests/test_attention.py -vv`
- Exit code: `0`
- Duration: `3.41622` seconds
- stdout: `01-correctness.stdout.txt` (`bac8e26b54c21869baa21763c022c179594bafa2bcea25893f875948e4b0f199`)
- stderr: `01-correctness.stderr.txt` (`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`)

### 02-benchmark

- Command: `.venv/Scripts/python.exe attention.py`
- Exit code: `0`
- Duration: `8.249732` seconds
- stdout: `02-benchmark.stdout.txt` (`3652e8150e08b9d479b130ebc9f232b9f2a348a61250acea7fba50f0fc2c6c8b`)
- stderr: `02-benchmark.stderr.txt` (`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`)


## Interpretation boundary

Passing tests establish the stated cases and tolerances, not universal correctness. Timing is a measurement on the recorded machine, software stack, shapes, and power state; it is not a portable speed guarantee.
