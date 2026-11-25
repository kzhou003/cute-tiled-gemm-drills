#!/usr/bin/env bash
# Run the acceptance tests for one drill (or all of them).
#   ./check.sh a1                 # test your mine/a1_*.py
#   ./check.sh a1 -x              # extra args go to pytest
#   ./check.sh all                # every drill
#   ./check.sh a1 --solutions     # run the reference solution instead (DRILL_IMPL=solutions)
#   ./check.sh oracle a1          # regenerate tests/golden/ for a1 with real CuTe (needs tools/fetch_headers.sh)
#   ./check.sh oracle all
set -euo pipefail
cd "$(dirname "$0")"

oracle_one() {
  case "$1" in
    c1) oracle/run.sh c1 && oracle/c1_asm.sh ;;
    c2) oracle/c2_try_compile.sh ;;
    d3|d4|e1|e2) echo "$1: no CuTe oracle (protocol / timeline simulator written from the CUTLASS source)" ;;
    *) oracle/run.sh "$1" ;;
  esac
}
if [[ "${1:-}" == "oracle" ]]; then
  which="${2:?usage: ./check.sh oracle <id|all>}"
  if [[ "$which" == "all" ]]; then
    for i in a1 a2 a3 a4 a5 b1 b2 c1 c2 d1 d2; do oracle_one "$i"; done
  else
    oracle_one "$which"
  fi
  exit 0
fi

id="${1:-all}"
shift || true
args=()
for a in "$@"; do
  if [[ "$a" == "--solutions" ]]; then export DRILL_IMPL=solutions; else args+=("$a"); fi
done

if [[ "$id" == "all" ]]; then
  targets=(tests)
else
  shopt -s nullglob
  targets=(tests/test_"${id}"_*.py)
  if (( ${#targets[@]} == 0 )); then
    echo "no test file matches tests/test_${id}_*.py" >&2
    exit 1
  fi
fi

if [[ -x .venv/bin/pytest ]]; then
  py=(.venv/bin/pytest)
elif python3 -c "import pytest" 2>/dev/null; then
  py=(python3 -m pytest)
else
  echo "first run: creating .venv with pytest" >&2
  python3 -m venv .venv && .venv/bin/pip install -q pytest numpy
  py=(.venv/bin/pytest)
fi
exec "${py[@]}" -q "${targets[@]}" ${args[@]+"${args[@]}"}
