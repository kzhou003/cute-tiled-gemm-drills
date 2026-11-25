#!/usr/bin/env bash
# Build and run oracle/<stem>.cpp against real CuTe headers; write tests/golden/<stem>.json.
#   oracle/run.sh a1          (or: ./check.sh oracle a1)
set -euo pipefail
cd "$(dirname "$0")/.."
id="${1:?usage: oracle/run.sh <id>}"
src=$(ls oracle/"${id}"_*.cpp 2>/dev/null | head -1)
[[ -n "$src" ]] || { echo "no oracle/${id}_*.cpp" >&2; exit 1; }
stem=$(basename "$src" .cpp)
TP=third_party
[[ -d $TP/cutlass/include ]] || { echo "run tools/fetch_headers.sh first" >&2; exit 1; }
mkdir -p oracle/build
${CXX:-clang++} -std=c++17 -O1 -w \
  -I $TP/cutlass/include -I $TP/cccl/libcudacxx/include -I $TP/cccl/cub -I $TP/cccl/thrust -I $TP/cuda/include \
  -I oracle "$src" -o "oracle/build/$stem"
"oracle/build/$stem" > "tests/golden/$stem.json.tmp"
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "tests/golden/$stem.json.tmp"
mv "tests/golden/$stem.json.tmp" "tests/golden/$stem.json"
echo "wrote tests/golden/$stem.json"
