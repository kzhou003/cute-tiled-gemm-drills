"""F5 · 在 RTX 5090 上扫配置：tile、stage、warp 排列、split-K，并和 cuBLAS 对比 — reference solution.

GPU 部分在 `f5_config_sweep_splitk.cu`：同一个模板化的流水线 kernel（gpu/pipelined_mainloop.cuh）实例化成几种配置，
每种配置跑正确性和计时，再跑 cuBLAS 当基准。本文件是主机部分：**在上机之前**先用代价模型给这些配置排序，
上机之后再和实测比，看模型错在哪里（知识页“配置决策”第 8 步）。

配置（和 .cu 里的 run<...>(...) 一一对应）
    CONFIGS = [Cfg(name, BM, BN, BK, stages, warp_m, warp_n, splits), ...]    given

legal(cfg, M, N, K) -> (bool, reason)
- M % BM == 0、N % BN == 0、K % (splits·BK) == 0，否则 (False, "divisibility")
- smem_bytes(cfg) = stages·(BM + BN)·BK·2 ≤ 99·1024，否则 (False, "smem")
- 合法时返回 (True, "")

predict_us(cfg, M, N, K) -> float：一个故意简单的模型（常数见下）
- CTA 数 = (M/BM)·(N/BN)·splits；最忙的 SM 顺序跑 ceil(CTA 数 / 170) 个 CTA
- 每个 CTA 的 k-tile 数 kt = K / (splits·BK)；每个 k-tile = 2·BM·BN·BK / 512 + 150 clk（计算 + 固定开销）
- 主循环 = ceil(CTA/170) · (kt · 每 k-tile + 1500 clk 填充)
- 显存下限 = ((M + N)·K·2 + M·N·4) 字节 / 1792 GB/s
- split-K 额外开销（splits > 1）：C 清零 + 每个 split 一遍 FP32 原子加 ≈ (1 + splits)·M·N·4 字节 / 1792 GB/s
- 结果 = max(主循环, 显存下限) + split-K 开销；时钟按 2.41 GHz 换算成 µs
rank(M, N, K) -> [(name, splits, predicted_us), ...]：只含合法配置，按预测时间从小到大
best(M, N, K) -> Cfg

compare(measured) -> dict：measured 是 .cu 打印的 JSON 列表（含 "kernel"、"splits"、"ms"）
- 返回 {"pairs": [(name, splits, predicted_us, measured_us), ...], "spearman": 秩相关系数}
- 只统计两边都有的配置；少于 2 个时 spearman 为 None

常数：SMS = 170，CLK_GHZ = 2.41，TC_FLOP_PER_CLK = 512，OVERHEAD_CLK = 150，FILL_CLK = 1500，DRAM_GBPS = 1792
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SMS, CLK_GHZ, TC_FLOP_PER_CLK = 170, 2.41, 512
OVERHEAD_CLK, FILL_CLK, DRAM_GBPS = 150, 1500, 1792
SMEM_LIMIT = 99 * 1024


@dataclass(frozen=True)
class Cfg:
    name: str
    BM: int
    BN: int
    BK: int
    stages: int
    warp_m: int
    warp_n: int
    splits: int = 1


CONFIGS = [
    Cfg("128x128x64_s3_w2x4", 128, 128, 64, 3, 2, 4),
    Cfg("128x128x64_s2_w2x4", 128, 128, 64, 2, 2, 4),
    Cfg("128x64x64_s4_w2x2", 128, 64, 64, 4, 2, 2),
    Cfg("64x64x64_s4_w2x2", 64, 64, 64, 4, 2, 2),
    Cfg("128x256x64_s2_w2x4", 128, 256, 64, 2, 2, 4),
    *[Cfg("128x128x64_s3_w2x4_splitk", 128, 128, 64, 3, 2, 4, s) for s in (2, 4, 5, 8)],
    *[Cfg("64x64x64_s4_w2x2_splitk", 64, 64, 64, 4, 2, 2, s) for s in (2, 4)],
]


def smem_bytes(cfg: Cfg) -> int:
    return cfg.stages * (cfg.BM + cfg.BN) * cfg.BK * 2


def legal(cfg: Cfg, M: int, N: int, K: int) -> tuple:
    if M % cfg.BM or N % cfg.BN or K % (cfg.splits * cfg.BK):
        return False, "divisibility"
    if smem_bytes(cfg) > SMEM_LIMIT:
        return False, "smem"
    return True, ""


def predict_us(cfg: Cfg, M: int, N: int, K: int) -> float:
    ok, why = legal(cfg, M, N, K)
    if not ok:
        raise ValueError(f"{cfg.name} (splits={cfg.splits}) is illegal for {M}x{N}x{K}: {why}")
    ctas = (M // cfg.BM) * (N // cfg.BN) * cfg.splits
    kt = K // (cfg.splits * cfg.BK)
    per_k = 2 * cfg.BM * cfg.BN * cfg.BK / TC_FLOP_PER_CLK + OVERHEAD_CLK
    main_us = math.ceil(ctas / SMS) * (kt * per_k + FILL_CLK) / (CLK_GHZ * 1e3)
    dram_us = ((M + N) * K * 2 + M * N * 4) / (DRAM_GBPS * 1e3)
    split_us = (1 + cfg.splits) * M * N * 4 / (DRAM_GBPS * 1e3) if cfg.splits > 1 else 0.0
    return max(main_us, dram_us) + split_us


def rank(M: int, N: int, K: int) -> list:
    out = [(c.name, c.splits, predict_us(c, M, N, K)) for c in CONFIGS if legal(c, M, N, K)[0]]
    return sorted(out, key=lambda x: x[2])


def best(M: int, N: int, K: int) -> Cfg:
    name, splits, _ = rank(M, N, K)[0]
    return next(c for c in CONFIGS if c.name == name and c.splits == splits)


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for pos, i in enumerate(order):
        r[i] = float(pos)
    return r


def compare(measured: list) -> dict:
    pairs = []
    for m in measured:
        if m.get("skipped") or m.get("baseline") or "ms" not in m:
            continue
        cfg = next((c for c in CONFIGS if c.name == m["kernel"] and c.splits == m.get("splits", 1)), None)
        if cfg is None or not legal(cfg, m["M"], m["N"], m["K"])[0]:
            continue
        pairs.append((cfg.name, cfg.splits, predict_us(cfg, m["M"], m["N"], m["K"]), m["ms"] * 1e3))
    if len(pairs) < 2:
        return {"pairs": pairs, "spearman": None}
    rp, rm = _ranks([p[2] for p in pairs]), _ranks([p[3] for p in pairs])
    n = len(pairs)
    d2 = sum((a - b) ** 2 for a, b in zip(rp, rm))
    return {"pairs": pairs, "spearman": 1 - 6 * d2 / (n * (n * n - 1))}
