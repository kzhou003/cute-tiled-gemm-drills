"""F4 · 边界和 epilogue：M、N 不整除时的谓词加载，α·acc + β·C，FP32 → BF16 写回 — your implementation (run ./check.sh f4).

GPU 部分在 `f4_epilogue_residue.cu`。本文件是主机部分：用 NumPy 复刻 kernel 的边界逻辑和 epilogue 的数值，
在没有 GPU 的机器上先把“哪些加载要关掉、哪些输出要写、写出来的 BF16 是多少”确认清楚。

边界（CTA tile 128×128×64，256 线程，加载布局同 F2：线程 t 的 4 条 cp.async 起点行是 t // 8 + 32·i）
- cta_grid(M, N) = (ceil(M / 128), ceil(N / 128))：最后一块不满也要一个 CTA。
- load_predicates(bm, t, M)：CTA 第 bm 行块里，线程 t 的 4 条 A 加载各自要不要发——全局行号 128·bm + t // 8 + 32·i < M。
  （B 完全一样，换成 bn 和 N。K 方向要求整除，所以谓词只看行。）
- store_mask(bm, bn, M, N)：(128,128) 的布尔数组，True = 这个输出在界内、要写回。
- tile_waste(M, N)：因为向上取整，所有 CTA 里白算的输出占比 = 1 − M·N / (CTA 数 · 128 · 128)。

为什么越界的 SMEM 不用清零：A 的第 m 行只参与 C 的第 m 行，B 的第 n 行只参与 C 的第 n 列。越界的行 / 列算出来的
累加器（哪怕是垃圾、甚至 NaN）只落在 store_mask 为 False 的位置，根本不会被写回。K 方向不一样：K 的越界会进入每一个
输出，所以 F4 仍然要求 K % 64 == 0。

epilogue 数值
- bf16_round(x)：FP32 → BF16，round-to-nearest-even（和 __float2bfloat16 / cute::bfloat16_t(float) 相同）。
  做法：看 FP32 的位模式 u，加上 0x7FFF + ((u >> 16) & 1) 再截掉低 16 位。NaN 保持 NaN。返回 float32 数组（值都是 BF16 能表示的）。
- epilogue(acc, c, alpha, beta)：D = bf16_round(alpha·acc + beta·c)；beta == 0 时**不读 c**（c 可以是任意值，包括 NaN）。
  acc、c 都是 float32 数组（c 是 BF16 值的 float32 表示）。
- max_rel_err(got, ref)：max|got − ref| / max(1, max|ref|)，和 gpu/common.cuh 的判定一致；BF16 输出的容差是 1e-2。

API
    BM, BN, BK, THREADS                                     given
    cta_grid(M, N) -> (int, int)
    load_predicates(bm, t, M) -> [bool] * 4
    store_mask(bm, bn, M, N) -> np.ndarray (128,128) bool
    tile_waste(M, N) -> float
    bf16_round(x) -> np.ndarray float32
    epilogue(acc, c, alpha, beta) -> np.ndarray float32
    max_rel_err(got, ref) -> float
"""
from __future__ import annotations

import numpy as np

BM, BN, BK, THREADS = 128, 128, 64, 256


def cta_grid(M: int, N: int) -> tuple:
    raise NotImplementedError("cta_grid")


def load_predicates(bm: int, t: int, M: int) -> list:
    raise NotImplementedError("load_predicates")


def store_mask(bm: int, bn: int, M: int, N: int) -> np.ndarray:
    raise NotImplementedError("store_mask")


def tile_waste(M: int, N: int) -> float:
    raise NotImplementedError("tile_waste")


def bf16_round(x) -> np.ndarray:
    raise NotImplementedError("bf16_round")


def epilogue(acc, c, alpha: float, beta: float) -> np.ndarray:
    raise NotImplementedError("epilogue")


def max_rel_err(got, ref) -> float:
    raise NotImplementedError("max_rel_err")
