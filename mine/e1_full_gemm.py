"""E1 · 完整成熟的 Tiled GEMM：TMA box → swizzled SMEM 环形缓冲 → mbarrier 流水线 → warp specialization → MMA → epilogue → PDL 串联 — your implementation (run ./check.sh e1).

这是 capstone：把前面所有部件接成一个 Hopper 风格的 warp-specialized GEMM，功能上逐元素正确，时间上能看到
搬运与计算的重叠，最后用 PDL 串联第二个 GEMM。结构对照 CUTLASS 的
include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp（1 个 producer warp group 发 TMA，
2 个 consumer warp group 做 MMA）、include/cutlass/pipeline/sm90_pipeline.hpp（PipelineTmaAsync、PipelineState）
和 examples/cute/tutorial/hopper/wgmma_tma_sm90.cu（mbarrier + TMA 的最小主循环）。

问题：C[M,N] = A[M,K] · B[N,K]ᵀ（A、B 都是 K 连续，即文档里的 “A × Bᵀ”）。每个 CTA 算一个 BM×BN 的 C tile，
沿 K 每次取 BK=64（一个 k-tile）。两个 consumer warp group 合作：WG w 负责 tile 的第 [w·BM/2, (w+1)·BM/2) 行。
数值用 float64 模拟（half 的字节数只用来算时间和 TMA 字节数：ELEM_BYTES = 2）。

given（不需要你写）
- Cfg(M, N, K, BM=128, BN=128, BK=64, stages=4)，以及时间常量 TMA_LAT_US、TMA_BYTES_PER_US、MMA_US、EPI_US、SETUP_US、
  STREAM_GAP_US、PDL_LAUNCH_US、FLUSH_US。
- sw128(offset)：16-bit 元素的 SW128 swizzle（CuTe Swizzle<3,3,3>，作用在元素偏移上）：o^ ((o >> 3) & 0x38)。
  一个 stage 的 A tile（BM×BK，K-major，BK = 64）里元素 (m, k) 放在 smem[stage_base + sw128(m·BK + k)]；B 同理（BN×BK）。
- smem_read(smem, base, rows, cols)：按同样的 swizzle 读出逻辑 tile（consumer 的视角）。
- Barrier：mbarrier 模型。init 时给出每个 phase 需要的 arrival 数；arrive(t)、expect_tx(bytes, t)、complete_tx(bytes, t)；
  phase 在“arrival 到齐且 tx 字节清零”时完成，完成时刻是参与事件时间的最大值。
  try_wait(parity, t) 与 PTX mbarrier.try_wait.parity 相同：截至时刻 t 已完成的 phase数为 c，c % 2 != parity 时通过。
- Sim：离散事件模拟器。每个 agent 是一个 generator，yield 下面的操作：
    ("wait", bar, parity)                 阻塞直到 bar.try_wait(parity) 通过
    ("arrive", bar)                       到达一次
    ("arrive_expect_tx", bar, nbytes)     到达一次并声明本 phase 还要等 nbytes 的 TMA字节
    ("tma", bar, nbytes, fn, stage)       发起 TMA：引擎串行传输 nbytes/TMA_BYTES_PER_US，传完再过 TMA_LAT_US 落地；
                                          落地时对 bar 做 complete_tx(nbytes)；fn()在发起时立刻执行（做数据搬运）
    ("compute", us, fn, stage)            占用 us 微秒后执行 fn()（stage 是读的 smem stage，不读写时为 None）
    ("grid_wait",)                        griddepcontrol.wait：等到 sim.grid_ready（前驱 kernel 结束 + FLUSH）
    ("trigger",)                          launch_dependents：记录 sim.trigger_time（只影响后继的启动时间）
  Sim 检查数据竞争：TMA 写某个 stage 的区间和对它的读区间重叠 → RaceError("WAR …")；读一个stage 时它最近一次写还没
  落地 → RaceError("RAW …")；所有 agent 都阻塞 → Deadlock。run() 返回每个 agent 的结束时间，并记录 tma / mma 忙区间。

你要实现
    PipelineState(stages, index=0, phase=0, count=0)
        .advance()                         index+1；回到 0 时 phase ^= 1；count+1（与CUTLASS PipelineState::operator++ 相同，stages == 1 时只翻 phase）
        PipelineState.producer_start(stages) -> PipelineState(stages, 0, 1, 0)   生产者从相反的 phase 开始，因为 buffer 一开始都是空的
        PipelineState.consumer_start(stages) -> PipelineState(stages, 0, 0, 0)
    tma_box_load(src, r0, c0, rows, cols, smem, base) -> int
        把 src[r0:r0+rows, c0:c0+cols] 按 sw128 写到 smem[base + …]；越界（r ≥ src 行数或 c ≥ 列数）的元素写 0（TMA 的
        OOB 填零）。返回传输的字节数 rows·cols·ELEM_BYTES（越界部分也计入，TMA box 总是整块）。
    producer(ctx)  generator：先 ("compute", SETUP_US, None, None)；
        对每个 k-tile：等 empty[s]（producer 的 phase）→ arrive_expect_tx(full[s], A和 B 两个 box 的字节和)
        → 两个 tma（先 A box 再 B box，stage 参数都传 s）→ state.advance()。
        ctx.pdl_wait 为 True 时（被 PDL 提前启动的后继 kernel）：前 min(stages, k_tiles)个 stage 先只发 B（权重，独立
        operand）的 TMA（expect_tx 仍然按 A+B 的总字节声明），然后 ("grid_wait",)，再补发这些 stage 的 A，之后正常循环
        （对应 CUTLASS example 63：在 griddepcontrol.wait 之前预取权重）。
    consumer(ctx, wg)  generator：先 ("compute", SETUP_US, None, None)；对每个 k-tile：等 full[s]（consumer 的 phase）→
        ("compute", MMA_US, fn, s)，fn 把 smem 里本 WG 负责的 A 行和整块 B 读出来，acc[wg] += A_part · B_partᵀ
        → arrive(empty[s]) → advance。最后一个 k-tile 之后：wg == 0 时 yield ("trigger",)（CUTLASS 在最后一个 tile 的
        mma_tail 之后调用 launch_dependent_grids），然后 ("compute", EPI_US, fn, None)把 acc 写回 C（裁掉越界部分）。
    run_gemm(A, B, cfg, start=0.0, grid_ready=0.0, pdl_wait=False) -> (C, report)
        对每个 CTA（tile_m 外层、tile_n 内层）建 smem、full/empty 两组 Barrier（full 每phase 1 个 arrival，
        empty 每 phase 2 个 arrival = consumer WG 数）、一个 Sim（起始时间 start、grid_ready），跑 producer + 2 个 consumer。
        假设每个 CTA 独占一个 SM、同时运行。report = {"end": 所有 CTA 结束时间的最大值, "trigger": 所有 CTA trigger
        时间的最大值, "tma_busy": CTA 0 的 TMA 忙时间, "mma_busy": CTA 0 的 MMA 忙时间（两个 WG 的区间取并集）,
        "overlap": CTA 0 里 MMA 忙区间与 TMA 忙区间交集的长度}。
    chain(X, W1, W2, cfg1, cfg2, pdl) -> (Y, spans)
        H = X·W1ᵀ，Y = H·W2ᵀ（第二个 GEMM 的 A 是 H，B 是 W2）。不用 PDL：第二个 kernel start = end1 + STREAM_GAP_US，
        grid_ready = start（不需要等）。用 PDL：start = trigger1 + PDL_LAUNCH_US，grid_ready = end1 + FLUSH_US，
        pdl_wait=True。spans = [(0, end1), (start2, end2)]。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ELEM_BYTES = 2            # half：只用于字节数与时间
TMA_LAT_US = 0.5          # TMA 传完到数据落地 / barrier 完成的延迟
TMA_BYTES_PER_US = 32768  # 每个 SM 的 TMA 传输速度：一个 128×64 + 128×64 的 stage（32KB）需要 1 µs
MMA_US = 0.6              # 一个 consumer WG 做完一个 k-tile 的 MMA（两个 WG 并行）
EPI_US = 1.0              # epilogue：写回 C tile
SETUP_US = 0.5            # kernel 开始时的初始化（barrier init、算地址）
STREAM_GAP_US = 3.0       # 普通 stream：前一个 kernel 结束到下一个开始
PDL_LAUNCH_US = 0.5       # launch_dependents 之后到后继 kernel 开始
FLUSH_US = 0.5            # 前驱结束到 griddepcontrol.wait 返回
NUM_CONSUMERS = 2


@dataclass(frozen=True)
class Cfg:
    M: int
    N: int
    K: int
    BM: int = 128
    BN: int = 128
    BK: int = 64
    stages: int = 4


def sw128(o):
    """CuTe Swizzle<3,3,3> 作用在 16-bit 元素偏移上（SW128）：把行号的低 3 位异或进 16 字节块号。"""
    return o ^ ((o >> 3) & 0x38)


def smem_read(smem, base, rows, cols):
    """读出 smem 里一个 rows×cols（K-major，cols == 64）的逻辑 tile。"""
    r = np.arange(rows)[:, None]
    c = np.arange(cols)[None, :]
    return smem[base + sw128(r * cols + c)]


class RaceError(RuntimeError):
    pass


class Deadlock(RuntimeError):
    pass


class Barrier:
    """mbarrier 模型：每个 phase 需要 count 次 arrival，且 expect_tx 声明的字节全部 complete_tx。"""

    def __init__(self, count: int):
        self.count = count
        self.completions = []      # 每个已完成 phase 的完成时刻
        self._arr = 0
        self._tx = 0
        self._t = 0.0

    def _maybe_complete(self):
        if self._arr == self.count and self._tx == 0:
            self.completions.append(self._t)
            self._arr, self._tx, self._t = 0, 0, 0.0

    def arrive(self, t):
        self._arr += 1
        self._t = max(self._t, t)
        self._maybe_complete()

    def expect_tx(self, nbytes, t):
        self._tx += nbytes
        self._t = max(self._t, t)

    def complete_tx(self, nbytes, t):
        self._tx -= nbytes
        self._t = max(self._t, t)
        self._maybe_complete()

    def try_wait(self, parity, t):
        c = sum(1 for x in self.completions if x <= t)
        return c % 2 != parity

    def next_pass_time(self, parity, t):
        """已知的完成时刻里，最早让 try_wait(parity) 通过的时刻；没有则 None。"""
        for x in sorted(self.completions):
            if x > t and self.try_wait(parity, x):
                return x
        return None


class Sim:
    """离散事件模拟器（见模块 docstring）。agents: list of (name, generator)。"""

    def __init__(self, start=0.0, grid_ready=0.0):
        self.start = start
        self.grid_ready = grid_ready
        self.trigger_time = None
        self.tma_free = start
        self.tma_busy = []          # (start, end) 传输区间
        self.mma_busy = []          # (start, end) consumer 计算区间（stage 不为 None的 compute）
        self.stage_ready = {}       # stage -> 最近一次 TMA 写落地时刻
        self.stage_reads = {}       # stage -> [(start, end)]
        self.stage_writes = {}      # stage -> [(start, end)]

    def run(self, agents):
        st = []
        for name, gen in agents:
            st.append({"name": name, "gen": gen, "t": self.start, "op": None, "done": False})
        for a in st:
            self._fetch(a)
        while True:
            live = [a for a in st if not a["done"]]
            if not live:
                break
            progressed = False
            for a in sorted(live, key=lambda a: a["t"]):
                if self._step(a):
                    progressed = True
                    break
            if not progressed:
                # 所有 agent 都卡在 wait 上，而且没有未来的完成事件能放行它们
                raise Deadlock("all agents blocked: " + ", ".join(f"{a['name']}@{a['op'][0]}" for a in live))
        return {a["name"]: a["t"] for a in st}

    def _fetch(self, a, value=None):
        try:
            a["op"] = a["gen"].send(value) if a["op"] is not None else next(a["gen"])
        except StopIteration:
            a["done"] = True
            a["op"] = None

    def _step(self, a):
        op = a["op"]
        kind = op[0]
        t = a["t"]
        if kind == "wait":
            _, bar, parity = op
            if not bar.try_wait(parity, t):
                nt = bar.next_pass_time(parity, t)
                if nt is None:
                    return False
                a["t"] = t = nt
        elif kind == "arrive":
            op[1].arrive(t)
        elif kind == "arrive_expect_tx":
            op[1].expect_tx(op[2], t)
            op[1].arrive(t)
        elif kind == "tma":
            _, bar, nbytes, fn, stage = op
            xs = max(t, self.tma_free)
            xe = xs + nbytes / TMA_BYTES_PER_US
            land = xe + TMA_LAT_US
            self.tma_free = xe
            self.tma_busy.append((xs, xe))
            for (rs, re) in self.stage_reads.get(stage, []):
                if re > xs and rs < land:
                    raise RaceError(f"WAR on stage {stage}: TMA write [{xs:.2f},{land:.2f}] overlaps read [{rs:.2f},{re:.2f}]")
            self.stage_writes.setdefault(stage, []).append((xs, land))
            self.stage_ready[stage] = max(self.stage_ready.get(stage, 0.0), land)
            if fn is not None:
                fn()
            bar.complete_tx(nbytes, land)
        elif kind == "compute":
            _, us, fn, stage = op
            if stage is not None:
                ready = self.stage_ready.get(stage)
                if ready is None or ready > t:
                    raise RaceError(f"RAW on stage {stage}: read at {t:.2f} before data lands at {ready}")
                for (ws, we) in self.stage_writes.get(stage, []):
                    if we > t and ws < t + us:
                        raise RaceError(f"WAR on stage {stage}: read [{t:.2f},{t+us:.2f}] overlaps TMA write [{ws:.2f},{we:.2f}]")
                self.stage_reads.setdefault(stage, []).append((t, t + us))
                self.mma_busy.append((t, t + us))
            a["t"] = t + us
            if fn is not None:
                fn()
        elif kind == "grid_wait":
            a["t"] = max(t, self.grid_ready)
        elif kind == "trigger":
            self.trigger_time = t if self.trigger_time is None else max(self.trigger_time, t)
        else:
            raise ValueError(f"unknown op {kind}")
        self._fetch(a, None)
        return True


def union_length(intervals):
    """区间并集的总长度。"""
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def intersect_length(a, b):
    """两组区间（各自取并集后）交集的总长度。"""

    def merge(iv):
        out = []
        for s, e in sorted(iv):
            if out and s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return out
    A, B = merge(a), merge(b)
    i = j = 0
    tot = 0.0
    while i < len(A) and j < len(B):
        lo = max(A[i][0], B[j][0])
        hi = min(A[i][1], B[j][1])
        if hi > lo:
            tot += hi - lo
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return tot


@dataclass
class Ctx:
    """一个 CTA 的运行上下文（run_gemm 负责构造）。"""
    cfg: Cfg
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    m0: int
    n0: int
    smemA: np.ndarray
    smemB: np.ndarray
    full: list
    empty: list
    acc: list
    pdl_wait: bool = False

    @property
    def k_tiles(self):
        return -(-self.cfg.K // self.cfg.BK)


# ----------------------------------------------------------------------------- your code

class PipelineState:
    """环形 stage 下标 + phase 位（对照 cutlass/pipeline/sm90_pipeline.hpp 的 PipelineState）。"""

    def __init__(self, stages, index=0, phase=0, count=0):
        raise NotImplementedError("PipelineState.__init__")

    def advance(self):
        """index+1；绕回 0 时翻转 phase；count+1。stages == 1 时只翻 phase。"""
        raise NotImplementedError("PipelineState.advance")

    @classmethod
    def producer_start(cls, stages):
        """生产者从 phase 1 开始：空 buffer 的 empty barrier 还没完成过任何 phase，wait(1) 立刻通过。"""
        raise NotImplementedError("PipelineState.producer_start")

    @classmethod
    def consumer_start(cls, stages):
        """消费者从 phase 0 开始：要等第一次 full 完成。"""
        raise NotImplementedError("PipelineState.consumer_start")


def tma_box_load(src, r0, c0, rows, cols, smem, base):
    """把 src[r0:r0+rows, c0:c0+cols] 按 sw128 写到 smem[base + …]，越界填 0，返回字节数。"""
    raise NotImplementedError("tma_box_load")


def producer(ctx):
    """producer warp：等 empty → expect_tx → 两个 TMA → advance。pdl_wait 时先预取 B再 grid_wait。"""
    raise NotImplementedError("producer")


def consumer(ctx, wg):
    """consumer warp group：等 full → MMA → arrive empty → advance；最后 trigger + epilogue。"""
    raise NotImplementedError("consumer")


def run_gemm(A, B, cfg, start=0.0, grid_ready=0.0, pdl_wait=False):
    """跑所有 CTA，返回 (C, report)。"""
    raise NotImplementedError("run_gemm")


def chain(X, W1, W2, cfg1, cfg2, pdl):
    """H = X·W1ᵀ，Y = H·W2ᵀ；返回 (Y, [(0, end1), (start2, end2)])。"""
    raise NotImplementedError("chain")
