"""D3 · Warp specialization：寄存器重分配预算、Cooperative 与 Ping-pong 两种调度的时间线— reference solution.

Hopper 的 CUTLASS GEMM（include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_{cooperative,pingpong}.hpp）
一个 CTA 有 3 个 warp group（WG，128 线程 = 4 个 warp）：
- WG0 = Producer：只发 TMA、跑调度器，几乎不需要寄存器，启动后立刻 `setmaxnreg.dec` 降到 40个；
- WG1、WG2 = Consumer0 / Consumer1：做 WGMMA（warp-group 级矩阵乘）和 epilogue，`setmaxnreg.inc` 升到 232 个，
  用来放累加器。两份 kernel 的常数完全相同：LoadRegisterRequirement = 40、MmaRegisterRequirement = 232
  （每线程寄存器 >= 208 的“重负载”配置用 24 / 240）。
区别在两个 Consumer 怎么分工：
- Cooperative：两个 WG 一起算同一个输出 tile（TiledMMA 用 256 线程），一起做 epilogue（各写一半）。
- Ping-pong：两个 WG 轮流各算一个完整 tile；一个有序屏障（OrderedSequenceBarrier<2,2>）让两个 WG 的
  MMA 严格交替、epilogue 也严格交替，于是一个 Consumer 写回 tile j 的时候，另一个正在用 Tensor Core 算 tile j+1。

寄存器模型
- 一个 SM 有 REGS_PER_SM = 65536 个 32 位寄存器。启动时每线程寄存器数
  launch_regs_per_thread(threads) = floor(65536 / threads / 8) * 8（寄存器以 8 个为单位分配）；384 线程 → 168。
- setmaxnreg 的目标值必须是 8 的倍数，且在 [24, 256] 内（validate_setmaxnreg，错误码 E_NREG_MULT8 / E_NREG_RANGE）。
- register_plan(load_wgs, load_regs, mma_wgs, mma_regs)：threads = 128 * (load_wgs+ mma_wgs)；
  launch_total = launch_regs * threads；final_total = 128*(load_wgs*load_regs + mma_wgs*mma_regs)；
  freed = 128*load_wgs*(launch_regs − load_regs)；needed = 128*mma_wgs*(mma_regs −launch_regs)；
  ok 当且仅当两个目标值都合法、load_regs <= launch_regs <= mma_regs，且 final_total <= launch_total。
  返回 dict：threads, launch_regs, launch_total, final_total, freed, needed, ok, errors（列表，错误码开头：
  E_NREG_MULT8、E_NREG_RANGE、E_DEALLOC_ABOVE_LAUNCH、E_ALLOC_BELOW_LAUNCH、E_BUDGET）。
- accumulator_regs(tile_m, tile_n, threads)：fp32 累加器每元素一个寄存器，均分给参与的线程：tile_m*tile_n/threads。

时间模型（单位任意；一个持久化 CTA 依次处理 n_tiles 个输出 tile；生产者的 TMA 假设总能跟上）
- mainloop：算完一个 tile 的全部 MMA 需要的 Tensor Core 时间（Tensor Core 是 SM 内共享的，两个 WG 一起算也不会更快）。
- epilogue：一个 WG 写回一整个 tile 的时间（Cooperative 两个 WG 各写一半，用 epilogue/2）。
- schedule_cooperative：tile j 的 MMA 在 tile j−1 的 epilogue 结束后开始；MMA 段 [s, s+M]，随后 epilogue 段 E/2，
  两个 WG 同时出现在同一段里（segments 里各记一条）。
- schedule_pingpong：tile j 归 WG j%2。mma_start_j = max(mma_end_{j−1}, epi_end_{j−2})；
  epi_start_j = max(mma_end_j, epi_end_{j−1})；epi_end_j = epi_start_j + E（不存在的项按 0 算）。
- Schedule：segments（Segment(wg, kind, tile, start, end)，kind 为 "mma" 或 "epi"，按开始时间、再按 wg 排序）、
  makespan（最后一段的结束时刻）、tensor_core_busy = n_tiles*M、utilization = tensor_core_busy / makespan。
- steady_per_tile(kind, M, E)：稳态下每个 tile 的平均间隔。cooperative → M + E/2；
  pingpong → max(M, E)：MMA 链每 tile 要 M，epilogue 链每 tile 要 E，单个 WG 每两个 tile 要 M + E（不会更紧）。
  所以 E <= 2M 时 Ping-pong 不慢于 Cooperative，E > 2M 时 Cooperative 反而更快。

API
    REGS_PER_SM, THREADS_PER_WG, SETMAXNREG_MIN, SETMAXNREG_MAX, Segment, Schedule given
    launch_regs_per_thread(threads) -> int
    validate_setmaxnreg(count) -> list[str]
    register_plan(load_wgs=1, load_regs=40, mma_wgs=2, mma_regs=232) -> dict
    accumulator_regs(tile_m, tile_n, threads) -> int
    schedule_cooperative(n_tiles, mainloop, epilogue) -> Schedule
    schedule_pingpong(n_tiles, mainloop, epilogue) -> Schedule
    steady_per_tile(kind, mainloop, epilogue) -> float
"""
from __future__ import annotations

from dataclasses import dataclass, field

REGS_PER_SM = 65536
THREADS_PER_WG = 128
SETMAXNREG_MIN = 24
SETMAXNREG_MAX = 256


@dataclass(frozen=True)
class Segment:
    wg: int
    kind: str          # "mma" | "epi"
    tile: int
    start: float
    end: float


@dataclass
class Schedule:
    segments: list = field(default_factory=list)
    makespan: float = 0.0
    tensor_core_busy: float = 0.0
    utilization: float = 0.0


def launch_regs_per_thread(threads: int) -> int:
    # 硬件按 8 个寄存器一组分配；编译器给 __launch_bounds__(384, 1) 的 kernel 最多分到 168个。
    return (REGS_PER_SM // threads) // 8 * 8


def validate_setmaxnreg(count: int) -> list[str]:
    errs = []
    if count % 8:
        errs.append(f"E_NREG_MULT8: setmaxnreg {count} is not a multiple of 8")
    if not (SETMAXNREG_MIN <= count <= SETMAXNREG_MAX):
        errs.append(f"E_NREG_RANGE: setmaxnreg {count} not in [{SETMAXNREG_MIN}, {SETMAXNREG_MAX}]")
    return errs


def register_plan(load_wgs: int = 1, load_regs: int = 40, mma_wgs: int = 2, mma_regs: int = 232) -> dict:
    threads = THREADS_PER_WG * (load_wgs + mma_wgs)
    launch = launch_regs_per_thread(threads)
    launch_total = launch * threads
    final_total = THREADS_PER_WG * (load_wgs * load_regs + mma_wgs * mma_regs)
    freed = THREADS_PER_WG * load_wgs * (launch - load_regs)
    needed = THREADS_PER_WG * mma_wgs * (mma_regs - launch)
    errors = validate_setmaxnreg(load_regs) + validate_setmaxnreg(mma_regs)
    # dec 只能往下降，inc 只能往上升：方向反了 setmaxnreg 本身就是非法的。
    if load_regs > launch:
        errors.append(f"E_DEALLOC_ABOVE_LAUNCH: producer target {load_regs} > launch {launch}")
    if mma_regs < launch:
        errors.append(f"E_ALLOC_BELOW_LAUNCH: consumer target {mma_regs} < launch {launch}")
    # inc 拿到的寄存器来自 dec 释放进 CTA 池子的那部分：总量不能超过启动时分到的。
    if final_total > launch_total:
        errors.append(f"E_BUDGET: final {final_total} registers > launch allocation {launch_total}")
    return dict(threads=threads, launch_regs=launch, launch_total=launch_total, final_total=final_total,
                freed=freed, needed=needed, ok=not errors, errors=errors)


def accumulator_regs(tile_m: int, tile_n: int, threads: int) -> int:
    return tile_m * tile_n // threads


def _finish(segs: list, n_tiles: int, mainloop: float) -> Schedule:
    segs.sort(key=lambda s: (s.start, s.wg))
    makespan = max((s.end for s in segs), default=0.0)
    busy = n_tiles * mainloop
    return Schedule(segments=segs, makespan=makespan, tensor_core_busy=busy,
                    utilization=(busy / makespan) if makespan else 0.0)


def schedule_cooperative(n_tiles: int, mainloop: float, epilogue: float) -> Schedule:
    segs, t = [], 0.0
    for j in range(n_tiles):
        # 两个 WG 一起：MMA 段里 Tensor Core 忙，epilogue 段里 Tensor Core 闲着（这是 Cooperative 的代价）。
        for wg in (1, 2):
            segs.append(Segment(wg, "mma", j, t, t + mainloop))
        t += mainloop
        for wg in (1, 2):
            segs.append(Segment(wg, "epi", j, t, t + epilogue / 2))
        t += epilogue / 2
    return _finish(segs, n_tiles, mainloop)


def schedule_pingpong(n_tiles: int, mainloop: float, epilogue: float) -> Schedule:
    segs = []
    mma_end, epi_end = {}, {}
    for j in range(n_tiles):
        wg = 1 + j % 2
        # 有序屏障：MMA 必须排在上一个 tile 的 MMA 之后（Tensor Core 轮流用）；
        # 同一个 WG 还得先写完自己上一个 tile（j−2）。
        s = max(mma_end.get(j - 1, 0.0), epi_end.get(j - 2, 0.0))
        mma_end[j] = s + mainloop
        segs.append(Segment(wg, "mma", j, s, mma_end[j]))
        # epilogue 也严格交替：等自己的 MMA 完、也等上一个 tile 的 epilogue 完。
        e = max(mma_end[j], epi_end.get(j - 1, 0.0))
        epi_end[j] = e + epilogue
        segs.append(Segment(wg, "epi", j, e, epi_end[j]))
    return _finish(segs, n_tiles, mainloop)


def steady_per_tile(kind: str, mainloop: float, epilogue: float) -> float:
    if kind == "cooperative":
        return mainloop + epilogue / 2
    if kind == "pingpong":
        # 两条独立的有序链：MMA 链（Tensor Core）和 epilogue 链。每个 WG 自己的 (M + E) /2 永远不超过 max(M, E)。
        return max(mainloop, epilogue)
    raise ValueError(f"unknown schedule kind {kind!r}")
