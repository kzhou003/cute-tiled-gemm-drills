"""F1 · 第一个真实的 CuTe kernel：local_tile + local_partition + SMEM，用 CUDA core 算 — reference solution.

这一题有两部分：
1. **GPU 部分**（`f1_simt_cute_gemm.cu`）：用真实 CuTe 写一个 SIMT（FFMA）的 tiled GEMM，在 RTX 5090 上跑通、对拍、计时。
2. **主机部分**（本文件）：把 kernel 里每个 `local_partition` 的结果先用整数算出来——哪个线程搬哪几个元素、
   哪个线程累加哪 64 个输出——并和真实 CuTe 的输出（oracle/f1_simt_cute_gemm.cpp → tests/golden）对拍。
   这样在没有 GPU 的机器上也能确认自己对 kernel 的理解是对的。

配置（和 .cu 一致，对照 CUTLASS examples/cute/tutorial/sgemm_1.cu）
- A: M×K、B: N×K（都按 K 连续），C: M×N（按 N 连续），C = A·Bᵀ。
- CTA tile (BM, BN, BK) = (128, 128, 8)，256 线程。
- 搬运：ThrA = ThrB = (32,8):(8,1)——线程 t 在线程布局里的坐标是 (t // 8, t % 8)。
  local_partition(tile, ThrA, t) 把 (128,8) 的 tile 按 (32,8) 切，线程 t 拿每一块里的同一个位置：
  (t // 8 + 32·i, t % 8)，i = 0..3（按 i 的顺序）。
- 计算：ThrC = (16,16):(1,16)——线程 t 的坐标是 (t % 16, t // 16)。
  线程 t 累加 C tile 的 (t % 16 + 16·a, t // 16 + 16·b)，a, b = 0..7；顺序是 a 先变（列主序，index = a + 8·b）。
  它读 sA 的行 t % 16 + 16·a（Step<_1, X> 只按行投影），读 sB 的行 t // 16 + 16·b。

计数
- global_bytes(M, N, K)：所有 CTA 从全局内存读的字节数 = CTA 数 × k-tile 数 × (BM + BN)·BK·2。
  （BF16 每元素 2 字节；不考虑 L2 命中——这是“L2→SM”的流量。）
- ffma_per_thread(K)：每个线程做的 FFMA 次数 = 8·8·K。
- smem_loads_per_ffma()：gemm(tCsA, tCsB, tCrC) 每个 k 读 8 个 a + 8 个 b，做 64 次 FFMA → 16/64。

API
    BM, BN, BK, THREADS                                    given
    cta_grid(M, N) -> (int, int)                           M、N 不是 128 的倍数时 ValueError
    copy_coords(t) -> [(m, k), ...]                        长度 4
    c_coords(t) -> [(m, n), ...]                           长度 64，列主序
    a_rows(t) -> [m, ...]                                  长度 8：线程 t 计算时读的 sA 行
    b_rows(t) -> [n, ...]                                  长度 8
    global_bytes(M, N, K) -> int
    ffma_per_thread(K) -> int
    smem_loads_per_ffma() -> float
"""
from __future__ import annotations

BM, BN, BK, THREADS = 128, 128, 8, 256


def _check_t(t: int) -> None:
    if not 0 <= t < THREADS:
        raise ValueError(f"thread {t} out of range [0, {THREADS})")


def cta_grid(M: int, N: int) -> tuple:
    if M <= 0 or N <= 0 or M % BM or N % BN:
        raise ValueError(f"F1 needs M, N to be positive multiples of {BM}, got {M}x{N}")
    return M // BM, N // BN


def copy_coords(t: int) -> list:
    _check_t(t)
    # ThrA (32,8):(8,1) 的逆：t → (t // 8, t % 8)；tile (128,8) 被切成 4 块 (32,8)，沿 M 排
    return [(t // 8 + 32 * i, t % 8) for i in range(BM // 32)]


def c_coords(t: int) -> list:
    _check_t(t)
    # ThrC (16,16):(1,16) 的逆：t → (t % 16, t // 16)；(128,128) 被切成 8×8 块，列主序遍历
    return [(t % 16 + 16 * a, t // 16 + 16 * b) for b in range(8) for a in range(8)]


def a_rows(t: int) -> list:
    _check_t(t)
    return [t % 16 + 16 * a for a in range(8)]


def b_rows(t: int) -> list:
    _check_t(t)
    return [t // 16 + 16 * b for b in range(8)]


def global_bytes(M: int, N: int, K: int) -> int:
    gm, gn = cta_grid(M, N)
    if K <= 0 or K % BK:
        raise ValueError(f"K must be a positive multiple of {BK}")
    return gm * gn * (K // BK) * (BM + BN) * BK * 2


def ffma_per_thread(K: int) -> int:
    return 8 * 8 * K


def smem_loads_per_ffma() -> float:
    return (8 + 8) / (8 * 8)
