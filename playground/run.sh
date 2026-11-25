#!/usr/bin/env bash
# Compile and run one playground program against the real CuTe / CUTLASS headers (host only, no GPU).
#   playground/run.sh playground/a1_cta_tiling.cpp
# Edit the .cpp files freely: they are experiments, not tests.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
src="${1:?usage: playground/run.sh playground/<file>.cpp}"
TP="$root/third_party"
[[ -d $TP/cutlass/include ]] || { echo "run tools/fetch_headers.sh first" >&2; exit 1; }
mkdir -p "$root/playground/build"
out="$root/playground/build/$(basename "$src" .cpp)"
${CXX:-clang++} -std=c++17 -O1 -w \
  -I "$TP/cutlass/include" -I "$TP/cccl/libcudacxx/include" -I "$TP/cccl/cub" -I "$TP/cccl/thrust" \
  -I "$TP/cuda/include" -I "$root/oracle" "$src" -o "$out"
"$out"
