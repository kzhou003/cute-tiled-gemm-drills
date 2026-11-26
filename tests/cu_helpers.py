"""Helpers for the Track F tests: type-check and run mine/<stem>.cu (or solutions/ with DRILL_IMPL).

- typecheck(stem): tools/cu_drill.py check — clang's CUDA front end (-fsyntax-only) or `nvcc -c`;
  skipped when neither exists (e.g. a Mac without `brew install llvm`).
- run_gpu(stem, *args): build with nvcc for $CUTE_DRILL_ARCH (default sm_120) and run; returns the JSON
  lines the program prints. Skipped when there is no NVIDIA GPU / nvcc / CUTLASS checkout.
Run pytest with -s (./check.sh f3 -s) to see every JSON line.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "cu_drill.py"


def cu_file(stem: str) -> Path:
    impl = os.environ.get("DRILL_IMPL", "mine")
    return ROOT / impl / f"{stem}.cu"


def _tool(*args, timeout=900):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True, text=True, timeout=timeout)


def typecheck(stem: str) -> None:
    if not (ROOT / "third_party" / "cutlass" / "include").is_dir():
        pytest.skip("third_party/cutlass missing: run tools/fetch_headers.sh")
    r = _tool("check", cu_file(stem))
    if r.returncode == 2:
        pytest.skip("no clang++ with the NVPTX target and no nvcc: install LLVM (brew install llvm) or CUDA to type-check")
    assert r.returncode == 0, f"{cu_file(stem).name} does not compile:\n{r.stderr[-6000:]}"


def gpu_ready() -> str | None:
    """None if a GPU run is possible, else the reason to skip."""
    if not shutil.which("nvidia-smi"):
        return "no NVIDIA GPU (nvidia-smi not found)"
    r = _tool("env", timeout=60)
    env = json.loads(r.stdout) if r.returncode == 0 else {}
    if not env.get("nvcc"):
        return "nvcc not found (set CUDA_HOME)"
    if not env.get("cutlass"):
        return "third_party/cutlass missing: run tools/fetch_headers.sh"
    return None


def run_gpu(stem: str, *args) -> list[dict]:
    why = gpu_ready()
    if why:
        pytest.skip(why)
    r = _tool("run", cu_file(stem), *args)
    lines = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    for l in lines:
        print(json.dumps(l, ensure_ascii=False))
    assert r.returncode == 0 and lines, f"{cu_file(stem).name} failed:\n{r.stdout[-3000:]}\n{r.stderr[-6000:]}"
    return lines
