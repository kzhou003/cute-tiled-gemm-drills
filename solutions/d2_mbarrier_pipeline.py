"""D2 · mbarrier 多级流水线：phase 奇偶、expect_tx、PipelineState，以及会死锁 / 会读脏数据的写法 — reference solution.

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
        self.arrive_count = arrive_count
        self.pending = arrive_count
        self.tx = 0
        self.completed = 0

    def _maybe_complete(self) -> None:
        # phase 完成需要两件事同时满足：所有预期的 arrive 都到了，且所有预期的字节都到了。
        if self.pending == 0 and self.tx == 0:
            self.completed += 1
            self.pending = self.arrive_count

    def arrive(self, n: int = 1) -> None:
        if n > self.pending:
            raise ProtocolError(f"arrive({n}) but only {self.pending} arrivals pending")
        self.pending -= n
        self._maybe_complete()

    def expect_tx(self, nbytes: int) -> None:
        self.tx += nbytes

    def arrive_and_expect_tx(self, nbytes: int) -> None:
        # 先登记字节、再 arrive：否则 arrive 那一刻 tx 还是 0，phase 会提前完成。
        self.expect_tx(nbytes)
        self.arrive()

    def complete_tx(self, nbytes: int) -> None:
        self.tx -= nbytes
        self._maybe_complete()

    def try_wait(self, parity: int) -> bool:
        return (self.completed & 1) != parity


class PipelineState:
    def __init__(self, stages: int, index: int = 0, phase: int = 0, count: int =0):
        self.stages, self.index, self.phase, self.count = stages, index, phase, count

    def increment(self) -> None:
        if self.stages == 1:
            self.phase ^= 1
            self.count += 1
            return
        self.index += 1
        self.count += 1
        if self.index == self.stages:
            self.index = 0
            self.phase ^= 1

    def advance(self, n: int) -> "PipelineState":
        # 与 CUTLASS 的 advance 相同：跨过 stage 边界的次数为奇数时翻转 phase。
        wraps = (self.index + n) // self.stages
        if wraps % 2 == 1:
            self.phase ^= 1
        self.index = (self.index + n) % self.stages
        self.count += n
        return self

    def as_tuple(self) -> tuple:
        return (self.index, self.phase, self.count)


def producer_start_state(stages: int) -> PipelineState:
    return PipelineState(stages, 0, 1, 0)


def simulate(stages: int, k_tiles: int, load_latency: float, mma_time: float,
             tx_bytes: int = 1024, bug: str | None = None) -> SimResult:
    full = [MBarrier(1) for _ in range(stages)]
    empty = [MBarrier(1) for _ in range(stages)]
    stage_data = [None] * stages        # 哪一块已经落地在这个 stage
    stage_busy = [None] * stages        # 哪一块占着这个 stage、还没被 release
    res = SimResult(total=0.0, load_issue=[None] * k_tiles, load_done=[None] * k_tiles,
                    mma_start=[None] * k_tiles, mma_end=[None] * k_tiles)
    events: list = []                   # (time, seq, callback)
    seq = [0]
    now = [0.0]

    def schedule(t, fn):
        heapq.heappush(events, (t, seq[0], fn))
        seq[0] += 1

    def producer():
        st = PipelineState(stages, 0, 0 if bug == "producer_phase0" else 1, 0)
        for k in range(k_tiles):
            i, p = st.index, st.phase
            if bug != "skip_acquire":
                yield ("wait", lambda i=i, p=p: empty[i].try_wait(p), f"producer waits empty[{i}] parity {p}")
            if stage_busy[i] is not None:
                raise RaceError(f"TMA for tile {k} overwrites stage {i} still holding tile {stage_busy[i]}")
            full[i].arrive_and_expect_tx(tx_bytes)
            stage_busy[i] = k
            res.load_issue[k] = now[0]
            def land(i=i, k=k):
                stage_data[i] = k
                res.load_done[k] = now[0]
                full[i].complete_tx(tx_bytes)

            schedule(now[0] + load_latency, land)
            st.increment()

    def consumer():
        st = PipelineState(stages, 0, 1 if bug == "consumer_phase1" else 0, 0)
        for k in range(k_tiles):
            i, p = st.index, st.phase
            yield ("wait", lambda i=i, p=p: full[i].try_wait(p), f"consumer waits full[{i}] parity {p}")
            if stage_data[i] != k:
                raise RaceError(f"consumer reads stage {i} for tile {k} but it holds {stage_data[i]}")
            res.mma_start[k] = now[0]
            yield ("sleep", mma_time, None)
            res.mma_end[k] = now[0]
            if bug != "no_release":
                stage_busy[i] = None
                empty[i].arrive()
            st.increment()

    # 每个参与者：生成器、当前在等的东西
    actors = [{"name": "producer", "gen": producer(), "cmd": None, "done": False},
              {"name": "consumer", "gen": consumer(), "cmd": None, "done": False}]

    def step(a) -> bool:
        """尽可能推进一个参与者；有进展返回 True。"""
        progressed = False
        while not a["done"]:
            cmd = a["cmd"]
            if cmd is not None:
                kind, arg, _ = cmd
                if kind == "wait" and not arg():
                    return progressed
                if kind == "sleep":
                    return progressed          # 由事件唤醒
            try:
                nxt = next(a["gen"])
            except StopIteration:
                a["done"] = True
                return True
            progressed = True
            a["cmd"] = nxt
            if nxt[0] == "sleep":
                def wake(a=a):
                    a["cmd"] = None
                schedule(now[0] + nxt[1], wake)
                return progressed
            if nxt[0] == "wait" and nxt[1]():
                a["cmd"] = None
        return progressed

    while True:
        # 同一时刻反复推进，直到谁都动不了（生产者排在前面）
        while any(step(a) for a in actors):
            pass
        if all(a["done"] for a in actors):
            break
        if not events:
            waits = "; ".join(a["cmd"][2] for a in actors if not a["done"] and a["cmd"] and a["cmd"][2])
            raise DeadlockError(f"deadlock at t={now[0]}: {waits}")
        t = events[0][0]
        now[0] = t
        while events and events[0][0] == t:
            _, _, fn = heapq.heappop(events)
            fn()
    res.total = res.mma_end[-1] if k_tiles else 0.0
    if k_tiles > stages:
        res.steady_interval = (res.mma_end[-1] - res.mma_end[-1 - stages]) / stages
    return res


def predicted_interval(stages: int, load_latency: float, mma_time: float) -> float:
    return max(mma_time, (load_latency + mma_time) / stages)


def min_stages(load_latency: float, mma_time: float) -> int:
    return math.ceil(load_latency / mma_time) + 1
