"""D2 · mbarrier 多级流水线：phase 奇偶、expect_tx、PipelineState，以及会死锁 / 会读脏数据的写法 — your implementation (run ./check.sh d2).

Hopper 的 GEMM 主循环用 S 个 shared-memory stage 做“边搬边算”：生产者（发 TMA 的线程）往 stage i 搬第 k 块，
消费者（做 MMA 的 warp group）算完第 k−S 块后把 stage 让出来。同步靠每个 stage 的两个 mbarrier：
full[i]（数据到了）和 empty[i]（可以覆盖了）。CUTLASS 把协议封装在 include/cutlass/pipeline/sm90_pipeline.hpp
的 PipelineTmaAsync 和 PipelineState 里；本题复刻它们，并用离散事件模拟主循环的时间和错误。

mbarrier 模型（PTX ISA 的 mbarrier 语义，简化到本题需要的部分）
- init(arrive_count)：每个 phase 需要 arrive_count 次 arrive。phase 编号从 0 开始。
- arrive(n=1)：待到达数减 n；减到负数抛 ProtocolError。
- expect_tx(b)：待完成的 transaction 字节 += b；complete_tx(b)：-= b（TMA 数据落地时由硬件做）。
- 当“待到达数 == 0 且 tx == 0”时当前 phase 完成：完成计数 +1，待到达数重置为 arrive_count。
- try_wait(parity)：parity 指定的那个 phase 已经完成时返回 True。规则：当前（未完成）phase的奇偶
  是 completed % 2，所以 try_wait(parity) == (completed % 2 != parity)。新建的 barrier 上 try_wait(1) 立即为 True。

PipelineState（逐行对照 CUTLASS）
- 字段 index、phase、count；++：index+1、count+1，index 到 S 时回 0 并翻转 phase（S == 1时每次都翻转 phase）。
- advance(n) 与 n 次 ++ 等价（CUTLASS 用整除算翻转次数的奇偶）。
- 消费者起点 (0, 0, 0)；生产者起点 make_producer_start_state() = (0, 1, 0)：buffer 一开始是空的，
  所以生产者等 empty 的“上一个 phase”，在新 barrier 上立即通过。

协议（PipelineTmaAsync）
- producer_acquire(st)：等 empty[st.index].try_wait(st.phase)，然后 full[st.index].arrive_and_expect_tx(tx_bytes)。
- TMA 发出后经过 load_latency，硬件在 full[st.index] 上 complete_tx(tx_bytes)。
- consumer_wait(st)：等 full[st.index].try_wait(st.phase)；consumer_release(st)：empty[st.index].arrive()。

simulate(stages, k_tiles, load_latency, mma_time, tx_bytes=1024, bug=None)
- 生产者：每块 k：acquire → arrive_and_expect_tx → 在当前时刻发 TMA（发射不耗时）→ ++state。
- 消费者：每块 k：wait → 检查 stage 里确实是第 k 块 → 计算 mma_time → release → ++state。
- 同一时刻生产者先于消费者执行；时间只在“等不到”时推进到下一个事件。
- bug：None | "producer_phase0"（生产者起点 phase 用 0）| "consumer_phase1"（消费者起点phase 用 1）
       | "no_release"（消费者从不 release）| "skip_acquire"（生产者不等 empty）。
- 检测：没有可运行的参与者、也没有待发生的事件但还没跑完 → DeadlockError；
  消费者读到的 stage 不是第 k 块 → RaceError；生产者往一个还没被 release 的 stage 发 TMA →RaceError。
- 返回 SimResult：total（最后一块算完的时刻）、load_issue / load_done / mma_start / mma_end（每块一个时间）、
  steady_interval（最后 S 块的平均间隔 (mma_end[-1] − mma_end[-1−S]) / S；k_tiles <= S时为 None）。

公式
    predicted_interval(S, L, T) = max(T, (L + T) / S)     一个 stage 的周期是 L + T，S 个 stage 轮转
    min_stages(L, T)            = ceil(L / T) + 1         让 (L + T) / S <= T 的最小S

API
    ProtocolError, DeadlockError, RaceError, SimResult           given
    MBarrier(arrive_count): arrive / expect_tx / arrive_and_expect_tx / complete_tx/ try_wait / completed
    PipelineState(stages, index=0, phase=0, count=0): increment() / advance(n) / as_tuple()
    producer_start_state(stages) -> PipelineState
    simulate(...) -> SimResult
    predicted_interval(stages, load_latency, mma_time) -> float
    min_stages(load_latency, mma_time) -> int
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field


class ProtocolError(RuntimeError):
    """mbarrier 被错误使用（例如 arrive 次数超过预期）。"""


class DeadlockError(RuntimeError):
    """所有参与者都在等，且不会再有事件发生。"""


class RaceError(RuntimeError):
    """读到还没落地的数据，或覆盖了还没被消费的 stage。"""


@dataclass
class SimResult:
    total: float
    load_issue: list = field(default_factory=list)
    load_done: list = field(default_factory=list)
    mma_start: list = field(default_factory=list)
    mma_end: list = field(default_factory=list)
    steady_interval: float | None = None


class MBarrier:
    def __init__(self, arrive_count: int):
        raise NotImplementedError("MBarrier.__init__")


    def arrive(self, n: int = 1) -> None:
        raise NotImplementedError("MBarrier.arrive")

    def expect_tx(self, nbytes: int) -> None:
        raise NotImplementedError("MBarrier.expect_tx")

    def arrive_and_expect_tx(self, nbytes: int) -> None:
        # 先登记字节、再 arrive：否则 arrive 那一刻 tx 还是 0，phase 会提前完成。
        raise NotImplementedError("MBarrier.arrive_and_expect_tx")

    def complete_tx(self, nbytes: int) -> None:
        raise NotImplementedError("MBarrier.complete_tx")

    def try_wait(self, parity: int) -> bool:
        raise NotImplementedError("MBarrier.try_wait")


class PipelineState:
    def __init__(self, stages: int, index: int = 0, phase: int = 0, count: int =0):
        raise NotImplementedError("PipelineState.__init__")

    def increment(self) -> None:
        raise NotImplementedError("PipelineState.increment")

    def advance(self, n: int) -> "PipelineState":
        # 与 CUTLASS 的 advance 相同：跨过 stage 边界的次数为奇数时翻转 phase。
        raise NotImplementedError("PipelineState.advance")

    def as_tuple(self) -> tuple:
        raise NotImplementedError("PipelineState.as_tuple")


def producer_start_state(stages: int) -> PipelineState:
    raise NotImplementedError("producer_start_state")


def simulate(stages: int, k_tiles: int, load_latency: float, mma_time: float,
             tx_bytes: int = 1024, bug: str | None = None) -> SimResult:
    raise NotImplementedError("simulate")


def predicted_interval(stages: int, load_latency: float, mma_time: float) -> float:
    raise NotImplementedError("predicted_interval")


def min_stages(load_latency: float, mma_time: float) -> int:
    raise NotImplementedError("min_stages")
