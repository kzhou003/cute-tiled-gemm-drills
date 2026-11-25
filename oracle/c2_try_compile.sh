#!/usr/bin/env bash
# Try to compile every oracle/c2_snippets/*.cpp against the real CuTe headers (-fsyntax-only) and
# record whether it compiles and the first static_assert message. Writes tests/golden/c2_compile_checks.json.
set -euo pipefail
cd "$(dirname "$0")/.."
TP=third_party
[[ -d $TP/cutlass/include ]] || { echo "run tools/fetch_headers.sh first" >&2; exit 1; }
mkdir -p oracle/build/c2
for f in oracle/c2_snippets/*.cpp; do
  n=$(basename "$f" .cpp)
  if ${CXX:-clang++} -std=c++17 -fsyntax-only -w \
      -I $TP/cutlass/include -I $TP/cccl/libcudacxx/include -I $TP/cccl/cub -I $TP/cccl/thrust -I $TP/cuda/include \
      "$f" 2> "oracle/build/c2/$n.err"; then echo ok > "oracle/build/c2/$n.status"; else echo fail > "oracle/build/c2/$n.status"; fi
done
python3 - <<'PY'
import json, re, glob, os
out = {"source": "clang++ -fsyntax-only against the pinned CuTe headers", "snippets": {}}
for st in sorted(glob.glob("oracle/build/c2/*.status")):
    n = os.path.basename(st)[:-7]
    ok = open(st).read().strip() == "ok"
    err = open(st[:-7] + ".err").read()
    msg = ""
    m = re.search(r"static assertion failed(?: due to requirement '.*?')?: ([^\n]+)", err)
    if m:
        msg = m.group(1).strip()
    elif not ok:
        m = re.search(r"error: ([^\n]+)", err)
        msg = "(no static_assert) " + (m.group(1).strip() if m else "")
    out["snippets"][n] = {"compiles": ok, "message": msg}
    print(f"{n:22} {'ok  ' if ok else 'FAIL'} {msg}")
json.dump(out, open("tests/golden/c2_compile_checks.json", "w"), indent=1)
PY
