#!/usr/bin/env bash
# Compile oracle/c1_asm/index_fns.cpp with -O2 -S (host compiler) and count multiply / divide /
# remainder instructions per function. Writes oracle/build/index_fns.s and tests/golden/c1_asm.json.
set -euo pipefail
cd "$(dirname "$0")/.."
TP=third_party
mkdir -p oracle/build
arch=()
# Apple silicon: emit arm64 even when the shell runs under Rosetta
if [[ "$(uname)" == Darwin && "$(sysctl -n hw.optional.arm64 2>/dev/null)" == 1 ]]; then arch=(-arch arm64); fi
${CXX:-clang++} -std=c++17 -O2 -S -w ${arch[@]+"${arch[@]}"} \
  -I $TP/cutlass/include -I $TP/cccl/libcudacxx/include -I $TP/cccl/cub -I $TP/cccl/thrust -I $TP/cuda/include \
  oracle/c1_asm/index_fns.cpp -o oracle/build/index_fns.s
python3 - <<'PY'
import json, re
asm = open("oracle/build/index_fns.s").read()
arch = "x86-64" if "%rsp" in asm or "%rdi" in asm else "arm64"
out = {"arch": arch, "functions": {}}
for fn in ("offset_static", "offset_dynamic"):
    m = re.search(r"^_?%s:[^\n]*\n(.*?)\.cfi_endproc" % fn, asm, re.S | re.M)
    body = m.group(1) if m else ""
    ins = [l.split()[0] for l in body.splitlines() if l.strip() and not l.strip().startswith((".", ";", "//", "#", "Ltmp", "LBB")) and not l.strip().endswith(":")]
    muldiv = [i for i in ins if re.fullmatch(r"(mul|madd|msub|smull|umull|sdiv|udiv|imul|idiv|div|imull|idivl|cltd|cdq)\w*", i)]
    out["functions"][fn] = {"instructions": len(ins), "mul_div": len(muldiv), "mul_div_ops": sorted(set(muldiv)),
                            "body": [l.strip() for l in body.splitlines() if l.strip() and not l.strip().startswith((".", "#", ";"))]}
json.dump(out, open("tests/golden/c1_asm.json", "w"), indent=1)
for fn, d in out["functions"].items():
    print(fn, d["instructions"], "instructions,", d["mul_div"], "mul/div", d["mul_div_ops"])
PY
