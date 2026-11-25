"""E2 · 选配置：CTA tile、stage 数、warp group 安排与 swizzle，在硬件约束下用代价模型排序 —reference solution.

文档的结论之一：CuTe 不会自动搜索 tile、stage、warp 数和 swizzle；它提供合法性检查和代码生成，性能选择要靠
设计好的搜索空间 + 代价模型 + 实测。CUTLASS 的 CollectiveBuilder（include/cutlass/gemm/collective/builders/
sm90_gmma_builder.inl）用 StageCountAutoCarveout 按 shared memory 容量算 stage 数；kernel 的寄存器分配见
sm90_gemm_tma_warpspecialized_cooperative.hpp 里的 LoadRegisterRequirement = 40 / MmaRegisterRequirement = 232。
这道题把这些约束和一个简化的代价模型写出来，在小搜索空间里排序，并和常见的 CUTLASS 选择对照。

数据（given）
- GPU(sms, smem_per_cta, regs_per_sm, hbm_bytes_per_us, tensor_flops_per_us, tma_lat_us)：H100_SXM 是示例规格
  （132 个 SM、每 CTA 最多 232448 字节 SMEM、每 SM 65536 个寄存器、3.35 TB/s、989 TFLOP/s dense FP16）。
- Config(BM, BN, BK, stages, schedule, swizzle_bytes)：schedule ∈ {"cooperative", "pingpong"}；
  cooperative = 2 个 consumer WG 合作算一个 BM×BN tile（各算 BM/2 行）；pingpong = 2 个consumer WG 各算一个完整的
  BM×BN tile、轮流做 MMA 和 epilogue。两者都有 1 个 producer WG。swizzle_bytes ∈ {32, 64, 128}。
- 常量：ELEM_BYTES = 2（FP16/BF16 输入），ACC_BYTES = 4（FP32 累加），LOAD_REGS = 40，MMA_REGS = 232，
  ACC_REG_OVERHEAD = 40（MMA WG 里除累加器外的寄存器），EPI_SMEM = 8192（epilogue 用的 SMEM），
  BARRIER_SMEM_PER_STAGE = 16，MAX_BOX = 256（TMA box 每一维最多 256 个元素），
  SWIZZLE_MMA_FACTOR = {128: 1.0, 64: 1.25, 32: 1.5}（swizzle 越窄，MMA 从 SMEM 读 operand 时 bank 冲突越多，粗略放大 MMA 时间）。

你要实现
    violations(cfg, gpu) -> list[str]    按下面的顺序检查，返回违反的规则名（都满足时返回[]）：
        "wgmma_m"     cooperative 时 BM % 128 != 0；pingpong 时 BM % 64 != 0（每个 WG的 wgmma M = 64）
        "wgmma_n"     BN % 8 != 0 或 BN > 256
        "swizzle"     BK·ELEM_BYTES 不是 swizzle_bytes 的整数倍（K-major 的 swizzle atom 一行正好是 swizzle_bytes）
        "tma_box"     BM、BN、BK 中任何一个 > MAX_BOX
        "smem"        smem_bytes(cfg) > gpu.smem_per_cta
        "acc_regs"    acc_regs_per_thread(cfg) + ACC_REG_OVERHEAD > MMA_REGS
        "regs"        (128·LOAD_REGS + 2·128·MMA_REGS) > gpu.regs_per_sm
        "stages"      stages < 2
    smem_bytes(cfg) = stages·(BM + BN)·BK·ELEM_BYTES + EPI_SMEM + stages·BARRIER_SMEM_PER_STAGE
    acc_regs_per_thread(cfg)：累加器是 FP32，每个寄存器 4 字节存 1 个值。
        cooperative：BM·BN / (2·128)；pingpong：BM·BN / 128（每个 WG 持有整个 tile 的累加器）。
    estimate(cfg, gpu, M, N, K) -> dict     代价模型（时间单位 µs）：
        tiles = ceil(M/BM)·ceil(N/BN)；k_tiles = ceil(K/BK)；waves = ceil(tiles / sms)
        每个 k-tile：t_mma_k = 2·BM·BN·BK / (gpu.tensor_flops_per_us / sms) · SWIZZLE_MMA_FACTOR[swizzle_bytes]
                    t_load_k = (BM + BN)·BK·ELEM_BYTES / (gpu.hbm_bytes_per_us / sms_active)
                    其中 sms_active = min(tiles, sms)（同时在跑的 CTA 平分带宽），
                    再乘 L2 复用系数 reuse（见下）。
        L2 复用：一个 wave 里同时有 W = min(tiles, sms) 个 tile，按行主序排在 tiles_n = ceil(N/BN) 列的网格上，
                 distinct_m = ceil(W / tiles_n)，distinct_n = min(W, tiles_n)；
                 reuse = (distinct_m·BM + distinct_n·BN) / (W·(BM + BN))（只有不同的panel 才需要从 HBM 读）。
        流水线：step = max(t_mma_k, t_load_k, (gpu.tma_lat_us + t_load_k) / stages)（stage 不够时延迟藏不住）
        t_epi = BM·BN·ELEM_BYTES / (gpu.hbm_bytes_per_us / sms_active)（写回 C tile）
        cooperative：waves = ceil(tiles / sms)，t_wave = gpu.tma_lat_us + t_load_k + k_tiles·step + t_epi
        pingpong：每个 SM 同时处理两个 tile（两个 consumer WG 各一个），waves = ceil(tiles / (2·sms))；两个 WG 的 MMA
                 共用一个 SM 的 tensor core，不能重叠，但一个 WG 的 epilogue 可以和另一个的 MMA 重叠：
                 t_wave = max(2·k_tiles·t_mma_k, k_tiles·step) + gpu.tma_lat_us + t_load_k + t_epi
        total = waves·t_wave；tflops = 2·M·N·K / total / 1e6（单位 TFLOP/s：FLOP / µs / 1e6）
        返回 {"tiles", "waves", "wave_eff": tiles / (waves·sms·(2 if pingpong else1)), "step", "t_wave", "total", "tflops"}。
    search(space, gpu, M, N, K) -> list[(Config, estimate dict)]
        过滤掉 violations 非空的配置，按 total 升序排序；total 相同时按 (BM, BN, BK, stages, schedule, swizzle_bytes) 升序。
    default_space() -> list[Config]
        BM ∈ {64, 128, 256}，BN ∈ {64, 128, 256}，BK ∈ {32, 64, 128}，stages ∈ {2,3, 4, 5, 6, 7}，
        schedule ∈ {"cooperative", "pingpong"}，swizzle_bytes ∈ {32, 64, 128}，全组合。
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
ELEM_BYTES = 2
ACC_BYTES = 4
LOAD_REGS = 40
MMA_REGS = 232
ACC_REG_OVERHEAD = 40
EPI_SMEM = 8192
BARRIER_SMEM_PER_STAGE = 16
MAX_BOX = 256
SWIZZLE_MMA_FACTOR = {128: 1.0, 64: 1.25, 32: 1.5}   # 窄 swizzle 让 MMA 读 SMEM 时有 bank 冲突（见 D2 的 wavefront 表）


@dataclass(frozen=True)
class GPU:
    sms: int
    smem_per_cta: int
    regs_per_sm: int
    hbm_bytes_per_us: float
    tensor_flops_per_us: float
    tma_lat_us: float


H100_SXM = GPU(sms=132, smem_per_cta=232448, regs_per_sm=65536,
               hbm_bytes_per_us=3.35e6, tensor_flops_per_us=989e6, tma_lat_us=0.6)


@dataclass(frozen=True, order=True)
class Config:
    BM: int
    BN: int
    BK: int
    stages: int
    schedule: str = "cooperative"
    swizzle_bytes: int = 128


def smem_bytes(cfg: Config) -> int:
    """multi-stage A/B ring + epilogue + 每个 stage 两个 mbarrier。"""
    return cfg.stages * (cfg.BM + cfg.BN) * cfg.BK * ELEM_BYTES + EPI_SMEM + cfg.stages * BARRIER_SMEM_PER_STAGE


def acc_regs_per_thread(cfg: Config) -> int:
    """每个 MMA 线程为 FP32 累加器占用的寄存器数。"""
    # cooperative：两个 WG（256 线程）分摊一个 tile；pingpong：一个 WG（128 线程）持有整个tile
    threads = 2 * 128 if cfg.schedule == "cooperative" else 128
    return cfg.BM * cfg.BN // threads


def violations(cfg: Config, gpu: GPU) -> list:
    """按 docstring 的顺序返回违反的规则。"""
    out = []
    if cfg.BM % (128 if cfg.schedule == "cooperative" else 64):
        out.append("wgmma_m")
    if cfg.BN % 8 or cfg.BN > 256:
        out.append("wgmma_n")
    if (cfg.BK * ELEM_BYTES) % cfg.swizzle_bytes:
        out.append("swizzle")
    if max(cfg.BM, cfg.BN, cfg.BK) > MAX_BOX:
        out.append("tma_box")
    if smem_bytes(cfg) > gpu.smem_per_cta:
        out.append("smem")
    if acc_regs_per_thread(cfg) + ACC_REG_OVERHEAD > MMA_REGS:
        out.append("acc_regs")
    if 128 * LOAD_REGS + 2 * 128 * MMA_REGS > gpu.regs_per_sm:
        out.append("regs")
    if cfg.stages < 2:
        out.append("stages")
    return out


def estimate(cfg: Config, gpu: GPU, M: int, N: int, K: int) -> dict:
    """见模块 docstring 的代价模型。"""
    tiles_m, tiles_n = math.ceil(M / cfg.BM), math.ceil(N / cfg.BN)
    tiles = tiles_m * tiles_n
    k_tiles = math.ceil(K / cfg.BK)
    pp = cfg.schedule == "pingpong"
    W = min(tiles, gpu.sms)
    sms_active = W
    distinct_m = math.ceil(W / tiles_n)
    distinct_n = min(W, tiles_n)
    reuse = (distinct_m * cfg.BM + distinct_n * cfg.BN) / (W * (cfg.BM + cfg.BN))
    bw = gpu.hbm_bytes_per_us / sms_active
    t_mma_k = 2 * cfg.BM * cfg.BN * cfg.BK / (gpu.tensor_flops_per_us / gpu.sms) *SWIZZLE_MMA_FACTOR[cfg.swizzle_bytes]
    t_load_k = (cfg.BM + cfg.BN) * cfg.BK * ELEM_BYTES / bw * reuse
    # 延迟 + 一个 stage 的传输要被 stages 个 buffer 分摊；不够时 step 被它主导
    step = max(t_mma_k, t_load_k, (gpu.tma_lat_us + t_load_k) / cfg.stages)
    t_epi = cfg.BM * cfg.BN * ELEM_BYTES / bw
    if pp:
        waves = math.ceil(tiles / (2 * gpu.sms))
        t_wave = max(2 * k_tiles * t_mma_k, k_tiles * step) + gpu.tma_lat_us + t_load_k + t_epi
    else:
        waves = math.ceil(tiles / gpu.sms)
        t_wave = gpu.tma_lat_us + t_load_k + k_tiles * step + t_epi
    total = waves * t_wave
    return {
        "tiles": tiles,
        "waves": waves,
        "wave_eff": tiles / (waves * gpu.sms * (2 if pp else 1)),
        "step": step,
        "t_wave": t_wave,
        "total": total,
        "tflops": 2 * M * N * K / total / 1e6,
    }


def search(space, gpu: GPU, M: int, N: int, K: int) -> list:
    """合法配置按 total 升序（并列按配置字段升序）。"""
    out = [(c, estimate(c, gpu, M, N, K)) for c in space if not violations(c, gpu)]
    out.sort(key=lambda ce: (ce[1]["total"], ce[0]))
    return out


def default_space() -> list:
    """全组合搜索空间。"""
    return [Config(bm, bn, bk, st, sch, sw)
            for bm, bn, bk, st, sch, sw in itertools.product(
                [64, 128, 256], [64, 128, 256], [32, 64, 128], [2, 3, 4, 5, 6, 7],
                ["cooperative", "pingpong"], [32, 64, 128])]
