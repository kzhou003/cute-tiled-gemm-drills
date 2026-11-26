"""F2 · 第一个 Tensor Core GEMM：TiledCopy（cp.async 16 B）+ swizzle 过的 SMEM + TiledMMA — your implementation (run ./check.sh f2).

GPU 部分在 `f2_tensorcore_gemm.cu`：五个映射（gpu/tiled_gemm_config.hpp 里的 Config<128,128,64,1,2,4>）变成真实的
cp.async + mma.sync kernel。本文件是主机部分：把 kernel 里三次 partition 的结果和 SMEM 地址用整数算出来，
和真实 CuTe 的输出（oracle/f2_tensorcore_gemm.cpp → tests/golden）对拍。

① Copy TV（G2SCopy）：线程排成 (32,8):(8,1)，每线程每次 (1,8) 个 BF16 = 16 B
- 线程 t 在线程布局里是 (t // 8, t % 8)；一轮覆盖 32 行 × 64 列，(128,64) 的 tile 要 4 轮（沿 M）。
- copy_starts(t)：4 条 cp.async 的起点 (t // 8 + 32·i, 8·(t % 8))，i = 0..3；每条向 K 方向连续 8 个元素。
- copy_coords(t)：展开成 32 个 (m, k)，顺序是“先 8 个值，再下一轮”（partition_S 的 (CPY, CPY_M, CPY_K) 顺序）。

② SMEM（SmemLayoutA = tile_to_shape(Swizzle<3,3,3> ∘ (8,(8,8)):(8,(1,64)), (128,64))）
- 一个 atom 是 8 行 × 64 列 = 512 个元素（1 KB）；atom 沿 M 堆叠：第 r 行在第 r // 8 个 atom 里。
- atom 内：块号 j = k // 8（16 B 一块），行 r' = r % 8，偏移 = 64·j + 8·(r' ⊕ j) + k % 8。
- smem_offset(r, k) = 512·(r // 8) + 64·j + 8·((r % 8) ^ j) + k % 8（元素偏移，乘 2 是字节）。
- bank_group(r, k) = (smem_offset(r, k)·2 // 16) mod 8：这个元素所在的 16 B bank 组。
  ldmatrix 读“8 行的同一块”时，这 8 个 bank 组应当两两不同。

③ MMA（TiledMma：2×4 个 warp 的 SM80_16x8x16，Tile<32,64,16>）在 (128,128) 的 C tile 上
- 线程 t：warp w = t // 32，lane = t % 32，wm = w % 2，wn = w // 2（warp 排列 (2,4) 列主序）。
- 一个 atom（16×8）里 lane 的 4 个累加器：行 lane // 4 和 lane // 4 + 8，列 2·(lane % 4) 和 +1。
- 一个 TiledMMA 的“自然覆盖”是 32×32（2 个 warp × 16 行，4 个 warp × 8 列）；在 128×128 上沿 M 重复 4 次（步长 32），
  沿 N 重复 4 次（步长 32）。
- mma_c_coords(t)：64 个 (m, n)，顺序 = partition_C 的 ((2,2), 4, 4)：
  先列 +0/+1，再行 +0/+8，再 M 重复，最后 N 重复。

计数
- smem_bytes(BM, BN, BK, stages) = stages·(BM + BN)·BK·2
- mma_per_ktile(BM, BN, BK) = (BM/16)·(BN/8)·(BK/16)：一个 CTA 每个 k-tile 发多少条 mma.sync

API
    BM, BN, BK, THREADS, WARP_M, WARP_N                     given
    copy_starts(t) -> [(m, k), ...]                         长度 4
    copy_coords(t) -> [(m, k), ...]                         长度 32
    smem_offset(r, k) -> int                                0 <= r < 128, 0 <= k < 64，否则 ValueError
    bank_group(r, k) -> int
    mma_c_coords(t) -> [(m, n), ...]                        长度 64
    smem_bytes(BM, BN, BK, stages) -> int
    mma_per_ktile(BM, BN, BK) -> int
"""
from __future__ import annotations

BM, BN, BK, THREADS, WARP_M, WARP_N = 128, 128, 64, 256, 2, 4


def _check_t(t: int) -> None:
    if not 0 <= t < THREADS:
        raise ValueError(f"thread {t} out of range [0, {THREADS})")


def copy_starts(t: int) -> list:
    raise NotImplementedError("copy_starts")


def copy_coords(t: int) -> list:
    raise NotImplementedError("copy_coords")


def smem_offset(r: int, k: int) -> int:
    raise NotImplementedError("smem_offset")


def bank_group(r: int, k: int) -> int:
    raise NotImplementedError("bank_group")


def mma_c_coords(t: int) -> list:
    raise NotImplementedError("mma_c_coords")


def smem_bytes(BM: int, BN: int, BK: int, stages: int) -> int:
    raise NotImplementedError("smem_bytes")


def mma_per_ktile(BM: int, BN: int, BK: int) -> int:
    raise NotImplementedError("mma_per_ktile")
