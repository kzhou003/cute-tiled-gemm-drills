#!/usr/bin/env bash
# Download the headers the oracles and playground programs compile against (host-only, no GPU):
#   third_party/cutlass   NVIDIA/cutlass at the commit every source link in the pages is pinned to
#   third_party/cccl      NVIDIA/cccl (libcudacxx / cub / thrust)
#   third_party/cuda      CUDA public headers (cudart, nvcc's crt/) from NVIDIA's redistributable archives
# Nothing here is committed (third_party/ is in .gitignore).
set -euo pipefail
cd "$(dirname "$0")/.."
CUTLASS_SHA=af951046cd9cac9301d8e3e90863fd8d7edc9fa9
CUDA_REDIST=https://developer.download.nvidia.com/compute/cuda/redist
mkdir -p third_party
if [[ ! -d third_party/cutlass ]]; then
  git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git third_party/cutlass
fi
git -C third_party/cutlass fetch -q origin "$CUTLASS_SHA" 2>/dev/null || true
git -C third_party/cutlass checkout -q "$CUTLASS_SHA"
if [[ ! -d third_party/cccl ]]; then
  git clone --depth 1 https://github.com/NVIDIA/cccl.git third_party/cccl
fi
if [[ ! -f third_party/cuda/include/cuda_runtime_api.h ]]; then
  mkdir -p third_party/cuda
  for p in cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz \
           cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz; do
    curl -sSL "$CUDA_REDIST/$p" | tar -xJ -C third_party/cuda --strip-components=1 '*/include/*'
  done
  # the cccl copy inside the redist is older than third_party/cccl; keep only one
  rm -rf third_party/cuda/include/cuda third_party/cuda/include/cub third_party/cuda/include/thrust
fi
echo "headers ready: cutlass @ $(git -C third_party/cutlass rev-parse --short HEAD), cccl @ $(git -C third_party/cccl rev-parse --short HEAD)"
