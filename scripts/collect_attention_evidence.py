#!/usr/bin/env python3
"""Create a provenance-complete correctness and benchmark receipt.

The collector deliberately runs correctness before performance and stops before
the benchmark if the tests fail. It records source hashes, the Git diff hash,
the relevant runtime versions, GPU metadata, exact commands, exit codes, raw
stdout/stderr, and hashes of every output log.

Examples (PowerShell):
    .\.venv\Scripts\python.exe scripts\collect_attention_evidence.py --metadata-only
    .\.venv\Scripts\python.exe scripts\collect_attention_evidence.py `
        --output-dir evidence\runs\20260822-attention
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evidence" / "runs" / "attention-latest"
EXCLUDED_FINGERPRINT_PREFIXES = ("evidence/runs/",)

INVARIANTS = (
    {
        "id": "ATTN-SEM-001",
        "claim": "Implements causal softmax(QK^T / sqrt(D))V with no dropout, bias, or padding mask.",
        "evidence": "attention.attn_fwd and attention.eager_attention; numerical parity tests",
    },
    {
        "id": "ATTN-NUM-001",
        "claim": "Matches an fp32 eager reference and torch SDPA at atol=5e-2, rtol=5e-2.",
        "evidence": "tests/test_attention.py::test_attention_matches_naive",
    },
    {
        "id": "ATTN-CAUSAL-001",
        "claim": "Changing the final key/value cannot change earlier query outputs.",
        "evidence": "tests/test_attention.py::test_attention_causality",
    },
    {
        "id": "ATTN-API-001",
        "claim": "Output preserves q shape, dtype, and CUDA device; unsupported layouts fail explicitly.",
        "evidence": "tests/test_attention.py output and input-contract assertions",
    },
    {
        "id": "ATTN-MEM-001",
        "claim": "The Triton kernel tiles K/V and never stores an N x N score tensor in global memory.",
        "evidence": "static inspection of attention.attn_fwd; this is an implementation property, not a profiler measurement",
    },
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def sanitize_remote(remote: str) -> str | None:
    """Remove URL userinfo so a receipt can never copy an embedded credential."""
    remote = remote.strip()
    if not remote:
        return None
    if "://" not in remote:
        return remote
    scheme, rest = remote.split("://", 1)
    authority, separator, suffix = rest.partition("/")
    if "@" in authority:
        authority = authority.rsplit("@", 1)[1]
    return f"{scheme}://{authority}{separator}{suffix}"


def repository_snapshot() -> dict[str, Any]:
    listed = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    paths = sorted(
        path
        for path in listed.split("\0")
        if path and not path.replace("\\", "/").startswith(EXCLUDED_FINGERPRINT_PREFIXES)
    )
    files: dict[str, str] = {}
    for relative in paths:
        candidate = ROOT / relative
        if candidate.is_file():
            files[relative.replace("\\", "/")] = sha256_file(candidate)

    canonical = "".join(f"{path}\0{digest}\n" for path, digest in files.items()).encode("utf-8")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", ".", ":(exclude)evidence/runs/**"],
        cwd=ROOT,
        capture_output=True,
    )
    if diff.returncode != 0:
        raise RuntimeError(diff.stderr.decode("utf-8", errors="replace").strip())

    status_lines = [
        line
        for line in git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
        if not line[3:].replace("\\", "/").startswith(EXCLUDED_FINGERPRINT_PREFIXES)
    ]
    return {
        "head": git("rev-parse", "HEAD").strip(),
        "branch": git("branch", "--show-current").strip(),
        "remote": sanitize_remote(git("config", "--get", "remote.origin.url", check=False)),
        "dirty": bool(status_lines),
        "status": status_lines,
        "head_diff_sha256": sha256_bytes(diff.stdout),
        "source_tree_sha256": sha256_bytes(canonical),
        "files": files,
    }


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def nvidia_smi_snapshot() -> dict[str, str] | None:
    fields = (
        "name,driver_version,pstate,temperature.gpu,power.draw,power.limit,"
        "clocks.current.sm,clocks.current.memory"
    )
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return {"error": result.stderr.strip()}
    return {"query": fields, "value": result.stdout.strip()}


def environment_snapshot() -> dict[str, Any]:
    import torch
    import triton

    cuda_available = torch.cuda.is_available()
    gpu: dict[str, Any] | None = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "device_count": torch.cuda.device_count(),
        }
    return {
        "platform": platform.platform(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": portable_path(Path(sys.executable).resolve()),
        },
        "packages": {
            name: package_version(name)
            for name in ("torch", "triton", "triton-windows", "cuda-python", "pytest", "numpy")
        },
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": cuda_available,
            "sdpa_backends": {
                "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
                "memory_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math_enabled": torch.backends.cuda.math_sdp_enabled(),
                "cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
            },
        },
        "triton": {"version": triton.__version__},
        "gpu": gpu,
        "nvidia_smi": nvidia_smi_snapshot(),
    }


def portable_path(path: Path) -> str:
    """Describe executables without embedding a developer's home-directory path."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def command_string(command: Sequence[str]) -> str:
    portable: list[str] = []
    for argument in command:
        candidate = Path(argument)
        portable.append(portable_path(candidate) if candidate.is_absolute() else argument)
    return subprocess.list2cmdline(portable) if os.name == "nt" else shlex.join(portable)


def run_and_log(label: str, command: Sequence[str], output_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    before = nvidia_smi_snapshot()
    start = time.perf_counter()
    result = subprocess.run(
        list(command),
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    elapsed = time.perf_counter() - start
    after = nvidia_smi_snapshot()

    stdout_path = output_dir / f"{label}.stdout.txt"
    stderr_path = output_dir / f"{label}.stderr.txt"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return {
        "label": label,
        "command": command_string(command),
        "started_at_utc": started.isoformat(),
        "elapsed_seconds": round(elapsed, 6),
        "exit_code": result.returncode,
        "stdout": stdout_path.relative_to(output_dir).as_posix(),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr": stderr_path.relative_to(output_dir).as_posix(),
        "stderr_sha256": sha256_file(stderr_path),
        "gpu_before": before,
        "gpu_after": after,
    }


def render_markdown(receipt: dict[str, Any]) -> str:
    repo = receipt["repository"]
    env = receipt["environment"]
    lines = [
        "# PicoKernels attention evidence receipt",
        "",
        f"Generated: `{receipt['generated_at_utc']}`",
        "",
        "## Source identity",
        "",
        f"- Git HEAD: `{repo['head']}`",
        f"- Branch: `{repo['branch'] or '(detached)'}`",
        f"- Dirty worktree: `{'yes' if repo['dirty'] else 'no'}`",
        f"- Diff SHA-256: `{repo['head_diff_sha256']}`",
        f"- Complete source-tree SHA-256: `{repo['source_tree_sha256']}`",
        f"- Source stable during run: `{'yes' if receipt['source_stable_during_run'] else 'no'}`",
        "",
        "## Environment",
        "",
        f"- Python: `{env['python']['version']}`",
        f"- PyTorch: `{env['torch']['version']}` (CUDA `{env['torch']['cuda_runtime']}`)",
        f"- Triton: `{env['triton']['version']}`",
        f"- GPU: `{(env['gpu'] or {}).get('name', 'unavailable')}`",
        "",
        "## Contract under test",
        "",
    ]
    lines.extend(f"- `{item['id']}` — {item['claim']}" for item in receipt["invariants"])
    lines.extend(["", "## Sequential commands", ""])
    if receipt["commands"]:
        for item in receipt["commands"]:
            lines.extend(
                [
                    f"### {item['label']}",
                    "",
                    f"- Command: `{item['command']}`",
                    f"- Exit code: `{item['exit_code']}`",
                    f"- Duration: `{item['elapsed_seconds']}` seconds",
                    f"- stdout: `{item['stdout']}` (`{item['stdout_sha256']}`)",
                    f"- stderr: `{item['stderr']}` (`{item['stderr_sha256']}`)",
                    "",
                ]
            )
    else:
        lines.append("Metadata-only run: correctness and benchmark commands were not executed.")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Passing tests establish the stated cases and tolerances, not universal correctness. "
            "Timing is a measurement on the recorded machine, software stack, shapes, and power state; "
            "it is not a portable speed guarantee.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="write source/environment fingerprints without running CUDA tests or benchmarks",
    )
    parser.add_argument("--skip-tests", action="store_true", help="skip the correctness gate")
    parser.add_argument("--skip-benchmark", action="store_true", help="skip the performance run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repository_snapshot(),
        "environment": environment_snapshot(),
        "protocol": {
            "order": ["correctness", "benchmark"],
            "fail_fast": True,
            "seed": 0,
            "attention_benchmark": {
                "warmup_ms": 25,
                "measurement_ms": 100,
                "statistic": "median",
                "provider_order": ["eager", "torch_sdpa", "triton"],
                "shapes": [
                    {"B": 4, "H": 8, "N": n, "D": 64, "dtype": "fp16"}
                    for n in (256, 512, 1024, 2048, 4096, 32768)
                ],
            },
        },
        "invariants": list(INVARIANTS),
        "commands": [],
    }

    exit_code = 0
    if not args.metadata_only:
        if not args.skip_tests:
            test_command = [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_attention.py",
                "-vv",
            ]
            test_result = run_and_log("01-correctness", test_command, output_dir)
            receipt["commands"].append(test_result)
            exit_code = test_result["exit_code"]

        if exit_code == 0 and not args.skip_benchmark:
            benchmark_result = run_and_log(
                "02-benchmark",
                [sys.executable, "attention.py"],
                output_dir,
            )
            receipt["commands"].append(benchmark_result)
            exit_code = benchmark_result["exit_code"]

    repository_after = repository_snapshot()
    receipt["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["source_after_run"] = {
        "head": repository_after["head"],
        "head_diff_sha256": repository_after["head_diff_sha256"],
        "source_tree_sha256": repository_after["source_tree_sha256"],
    }
    receipt["source_stable_during_run"] = all(
        receipt["repository"][key] == repository_after[key]
        for key in ("head", "head_diff_sha256", "source_tree_sha256")
    )
    if not receipt["source_stable_during_run"] and exit_code == 0:
        # A timing result cannot be bound to one source snapshot if the source
        # changed while the commands were running.
        exit_code = 3
    receipt["overall_exit_code"] = exit_code

    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(render_markdown(receipt), encoding="utf-8")
    print(f"receipt: {receipt_path}")
    print(f"source tree sha256: {receipt['repository']['source_tree_sha256']}")
    print(f"exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
