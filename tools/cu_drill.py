#!/usr/bin/env python3
"""Build, run and type-check the Track F CUDA programs (mine/f*.cu, solutions/f*.cu).

    python3 tools/cu_drill.py run   FILE.cu [M N K [iters [extra args...]]]   # nvcc → run → print JSON lines
    python3 tools/cu_drill.py build FILE.cu                                 # nvcc only, print the binary path
    python3 tools/cu_drill.py check FILE.cu                                 # type check, no GPU needed
    python3 tools/cu_drill.py env                                           # what was found

Environment
    CUTE_DRILL_ARCH   target architecture for nvcc (default sm_120 = RTX 5090; e.g. sm_86, sm_89, sm_90)
    CUDA_HOME         CUDA toolkit root (otherwise nvcc on PATH, then /usr/local/cuda)
    CUTE_DRILL_CLANG  a clang++ with the NVPTX target, for `check` on machines without nvcc
    CUBLAS_ROOT       directory with include/cublas_v2.h and lib/libcublas.so* (F5's baseline);
                      otherwise the CUDA toolkit, then the pip package nvidia-cublas (nvidia/cu13, nvidia/cublas)

`check` uses clang's CUDA front end (-fsyntax-only: every CuTe template is instantiated and every
static_assert is checked, but no machine code is generated) when a suitable clang exists, otherwise
`nvcc -c` to /dev/null. Exit status 0 = OK, 2 = no tool available (callers skip), 1 = compile error.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TP = ROOT / "third_party"
GPU = ROOT / "gpu"
BUILD = ROOT / "build" / "cu"


def arch() -> str:
    return os.environ.get("CUTE_DRILL_ARCH", "sm_120")


def find_nvcc() -> str | None:
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    cands = [Path(home) / "bin" / "nvcc"] if home else []
    on_path = shutil.which("nvcc")
    if on_path:
        cands.append(Path(on_path))
    cands.append(Path("/usr/local/cuda/bin/nvcc"))
    for c in cands:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def clang_has_nvptx(clang: str) -> bool:
    try:
        out = subprocess.run([clang, "-print-targets"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "nvptx" in out


def find_clang() -> str | None:
    cands = [os.environ.get("CUTE_DRILL_CLANG", "")]
    cands += ["/opt/homebrew/opt/llvm/bin/clang++", "/usr/local/opt/llvm/bin/clang++"]
    cands += [shutil.which(n) or "" for n in ("clang++", "clang++-20", "clang++-19", "clang++-18", "clang++-17")]
    for c in cands:
        if c and Path(c).is_file() and clang_has_nvptx(c):
            return c
    return None


def find_cublas() -> tuple[str, str] | None:
    """(include dir, library file) for cuBLAS, or None."""
    roots = []
    if os.environ.get("CUBLAS_ROOT"):
        roots.append(Path(os.environ["CUBLAS_ROOT"]))
    nvcc = find_nvcc()
    if nvcc:
        roots.append(Path(nvcc).resolve().parent.parent)
    try:
        import nvidia  # type: ignore  # pip wheels (torch installs them)
        for base in getattr(nvidia, "__path__", []):
            roots += [Path(base) / "cu13", Path(base) / "cublas", Path(base) / "cu12"]
    except ImportError:
        pass
    for r in roots:
        inc = r / "include" / "cublas_v2.h"
        libs = sorted(glob.glob(str(r / "lib*" / "libcublas.so*"))) + sorted(glob.glob(str(r / "lib*" / "x64" / "cublas.lib")))
        libs = [l for l in libs if "Lt" not in Path(l).name]
        if inc.is_file() and libs:
            return str(r / "include"), libs[0]
    return None


def includes() -> list[str]:
    if not (TP / "cutlass" / "include").is_dir():
        sys.exit("third_party/cutlass is missing: run tools/fetch_headers.sh first")
    return ["-I", str(TP / "cutlass" / "include"), "-I", str(GPU)]


def needs_cublas(src: Path) -> bool:
    return "cublas_v2.h" in src.read_text()


def nvcc_cmd(nvcc: str, src: Path, out: str, compile_only: bool = False) -> list[str]:
    cmd = [nvcc, "-std=c++17", "-O3", f"-arch={arch()}", "--expt-relaxed-constexpr", "-lineinfo",
           "-Xcudafe", "--diag_suppress=177", *includes()]
    blas = find_cublas() if needs_cublas(src) else None
    if blas:
        cmd += ["-I", blas[0]]
    cmd += ["-c" if compile_only else "", str(src), "-o", out]
    if blas and not compile_only:
        lib = Path(blas[1])
        cmd += [f"-L{lib.parent}", f"-Xlinker=-l:{lib.name}", f"-Xlinker=-rpath={lib.parent}"]
    return [c for c in cmd if c]


def build(src: Path) -> Path:
    nvcc = find_nvcc()
    if not nvcc:
        sys.exit("nvcc not found (set CUDA_HOME or put nvcc on PATH)")
    key = hashlib.sha1()
    for f in [src, *sorted(GPU.glob("*"))]:
        key.update(f.read_bytes())
    key.update(" ".join(nvcc_cmd(nvcc, src, "x")).encode())
    BUILD.mkdir(parents=True, exist_ok=True)
    exe = BUILD / f"{src.parent.name}_{src.stem}_{arch()}_{key.hexdigest()[:10]}"
    if not exe.exists():
        r = subprocess.run(nvcc_cmd(nvcc, src, str(exe)), capture_output=True, text=True)
        if r.returncode:
            sys.stderr.write(r.stdout + r.stderr)
            print(json.dumps({"ok": False, "error": "nvcc failed", "file": str(src)}))
            sys.exit(1)
    return exe


def check(src: Path) -> int:
    clang = find_clang()
    if clang:
        cuda_root = os.environ.get("CUDA_HOME") or (str(Path(find_nvcc()).resolve().parent.parent) if find_nvcc() else str(TP / "cuda"))
        # F1–F5 only use sm_80 instructions, so sm_80 is enough for a syntax/template check
        cmd = [clang, "-x", "cuda", "-std=c++17", "-fsyntax-only", "--cuda-gpu-arch=sm_80",
               f"--cuda-path={cuda_root}", "-nocudalib", "--no-cuda-version-check", "-Wno-unknown-cuda-version",
               "-w", *includes(), "-I", str(TP / "cuda" / "include"), str(src)]
        blas = find_cublas() if needs_cublas(src) else None
        if blas:
            cmd[1:1] = ["-I", blas[0]]
        tool = "clang"
    else:
        nvcc = find_nvcc()
        if not nvcc:
            print(json.dumps({"ok": None, "skipped": True, "reason": "no nvcc and no clang++ with the NVPTX target"}))
            return 2
        cmd = nvcc_cmd(nvcc, src, os.devnull, compile_only=True)
        tool = "nvcc"
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stdout + r.stderr)
    print(json.dumps({"ok": r.returncode == 0, "tool": tool, "file": str(src)}))
    return 0 if r.returncode == 0 else 1


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "build", "check", "env"):
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "env":
        blas = find_cublas()
        print(json.dumps({"arch": arch(), "nvcc": find_nvcc(), "clang_nvptx": find_clang(),
                          "cublas": blas, "cutlass": (TP / "cutlass" / "include").is_dir()}, indent=1))
        return 0
    src = Path(sys.argv[2]).resolve()
    if cmd == "check":
        return check(src)
    exe = build(src)
    if cmd == "build":
        print(exe)
        return 0
    r = subprocess.run([str(exe), *sys.argv[3:]])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
