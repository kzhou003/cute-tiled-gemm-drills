"""D4 · PDL：两个串联 GEMM 的提前启动、等待位置与正确性检查 — reference solution.

文档里的例子：Producer 计算 H = X·W1，Consumer 计算 Y = H·W2。Consumer 必须等 H 写完，但它读的 W2 和
Producer 无关。PDL（Programmatic Dependent Launch，Hopper 起的 CUDA 特性）让 Consumer在 Producer 结束之前
就开始执行“与 H 无关的部分”（预取 W2、初始化），等到真正要读 H 时再等待。

CUTLASS 里对应两条 device 端指令（include/cutlass/arch/grid_dependency_control.h）：
- launch_dependent_grids()  = griddepcontrol.launch_dependents：提示后继 kernel 可以开始启动。
  源码注释原话：它“不影响功能，只影响性能”——太早会和当前 kernel 抢资源，太晚就没有重叠。
- wait_on_dependent_grids() = griddepcontrol.wait：在它之前不得访问前驱 kernel 可能写的全局内存；
  它等待前驱 kernel **完全结束并把内存刷出**，不是只等到前驱的 trigger。
主机端要用 cudaLaunchAttributeProgrammaticStreamSerialization 发射（CUDA Graph 里则是programmatic 依赖边）。
CUDA runtime 里的同名 API：cudaTriggerProgrammaticLaunchCompletion() / cudaGridDependencySynchronize()。

模型（所有时间单位是微秒）
- Phase(name, us, reads, writes)：kernel 里顺序执行的一段，reads / writes 是全局 buffer名字的 frozenset。
- Kernel(name, phases, trigger_after, wait_before)：
    trigger_after：在第 trigger_after 个 phase **结束后** 调用 launch_dependents（None= 不调用，等价于
                   kernel 结束时才隐式触发）；
    wait_before：在第 wait_before 个 phase **开始前** 调用 griddepcontrol.wait（None =不等待）。
    0 ≤ trigger_after < len(phases)，0 ≤ wait_before ≤ len(phases)（= len 表示在所有phase 之后等待）。
- 常量：STREAM_GAP_US（普通 stream 里前一个 kernel 结束到下一个开始的空隙）、GRAPH_GAP_US（CUDA graph 节点之间）、
  PDL_LAUNCH_US（收到 launch_dependents 后到后继 kernel 开始执行的延迟）、FLUSH_US（前驱结束后内存刷出、
  wait 返回所需的时间）。

API
    Phase, Kernel, STREAM_GAP_US, GRAPH_GAP_US, PDL_LAUNCH_US, FLUSH_US    given
    two_gemm_chain(...)                                                     given：文档里的 H = X·W1 → Y = H·W2
    kernel_time(k) -> float              所有 phase 时长之和
    trigger_time(k) -> float             从 kernel 开始到 launch_dependents 的时间；trigger_after 为 None 时 = kernel_time(k)
    timeline(kernels, mode) -> dict      mode ∈ {"stream", "graph", "pdl", "graph_pdl"}；返回
        {"spans": [(start, end) 每个 kernel], "phases": [[(name, start, end), ...]每个 kernel], "total": 最后一个 end}
        - kernel 0 从 0 开始，phase 顺序执行；
        - stream / graph：start_i = end_{i-1} + gap，gap = STREAM_GAP_US / GRAPH_GAP_US；kernel 的 trigger / wait 被忽略；
        - pdl / graph_pdl（i ≥ 1）：如果前驱有 trigger，start_i = start_{i-1} + trigger_time(k_{i-1}) + PDL_LAUNCH_US；
          如果前驱没有 trigger，退化成 start_i = end_{i-1} + gap（gap 同上：pdl 用 STREAM_GAP_US，graph_pdl 用 GRAPH_GAP_US）。
          phase 顺序执行；第 wait_before 个 phase 的开始时间要和 end_{i-1} + FLUSH_US 取max（wait 返回的时刻）；
          wait_before == len(phases) 时 kernel 的结束时间和它取 max。wait_before 为 None 时不等待（可能不正确，见 check）。
        - contention（默认 0）：PDL 下后继 kernel 在 wait 之前执行的 phase 和前驱重叠时会抢资源（CUTLASS 注释：
          “launching a dependent kernel too early can compete with current kernels”）。模型：先按不重叠算出前驱的名义结束
          时间 E0，后继 wait 之前的那些 phase 覆盖的区间是 [start_i, w_i)（w_i = 第一个等待 phase 的名义开始时间，
          没有 wait 时 = 后继所有 phase 跑完的时间），重叠 o = max(0, min(E0, w_i) − start_i)，前驱的实际结束时间
          E = E0 + contention·o（只拉长前驱的结束；它的 trigger 时刻不变）。这个 E 用于 wait 和 spans，并作为下一对的“前驱结束”。
          stream / graph 模式下 contention 不起作用。
    best_trigger(kernels, mode="pdl", contention=0.0) -> (trigger_after, total)
                                         对 kernels[0] 枚举 trigger_after ∈ {0, …, len(phases)−1}，返回总时间最短的那个
                                         （并列时取更大的 trigger_after，也就是更晚的 trigger）。
    check(kernels) -> list[str]          PDL 下的数据竞争（对每一对相邻 kernel (p, c)检查），按下面的格式逐条返回，
                                         顺序 = (kernel 序号, phase 序号, buffer 名字排序)：
        "read-before-wait: <c.name>.<phase> reads <buf>"     c 在 wait 之前（或没有 wait 时任何位置）的 phase 读了 p 写的 buffer
        "write-before-wait: <c.name>.<phase> writes <buf>"   c 在 wait 之前写了 p 读或写的 buffer（WAR / WAW）
        没有问题时返回 []。trigger 的位置不影响检查结果（trigger 不保证可见性，正确性只靠 wait）。
    independent_operands(producer, consumer) -> list[str]
                                         consumer 读的 buffer 里 producer 不写的那些（排序）。它们可以在 wait 之前加载，
                                         对应 CUTLASS example 63 里在 griddepcontrol.wait 之前预取的权重。
    latest_wait(producer, consumer) -> int
                                         让 check 通过、且 wait 尽量晚的位置：consumer第一个“读 producer 写的 buffer 或
                                         写 producer 读写的 buffer”的 phase 序号；没有这样的 phase 时返回 len(phases)。
    speedup(kernels, base, opt) -> float timeline(base)["total"] / timeline(opt)["total"]
"""
from __future__ import annotations

from dataclasses import dataclass, replace

STREAM_GAP_US = 3.0    # 普通 stream：前一个 kernel 完全结束到下一个开始执行
GRAPH_GAP_US = 1.0     # CUDA graph：节点之间的空隙更小
PDL_LAUNCH_US = 0.5    # 收到 launch_dependents 之后，后继 kernel 开始执行的延迟
FLUSH_US = 0.5         # 前驱 kernel 结束后到 griddepcontrol.wait 返回（内存刷出）


@dataclass(frozen=True)
class Phase:
    """kernel 里顺序执行的一段。reads / writes 是全局 buffer 名字的 frozenset。"""
    name: str
    us: float
    reads: frozenset = frozenset()
    writes: frozenset = frozenset()


@dataclass(frozen=True)
class Kernel:
    """trigger_after：第几个 phase 结束后 launch_dependents（None = 不调用）；wait_before：第几个 phase 前 wait（None = 不等待）。"""
    name: str
    phases: tuple
    trigger_after: int | None = None
    wait_before: int | None = None


def two_gemm_chain(trigger_after_main: bool = True, consumer_wait: str = "mainloop"):
    """文档里的例子：producer H = X·W1，consumer Y = H·W2（时长是示意的微秒数）。

    producer：setup(2) → mainloop(20, 读 X W1) → epilogue(4, 写 H)；trigger 在 mainloop 之后（对应 CUTLASS 在最后一个
    tile 的 mainloop 结束、epilogue 之前调用 launch_dependent_grids），trigger_after_main=False 时不 trigger。
    consumer：setup(2) → prefetch_w2(5, 读 W2) → mainloop(15, 读 H W2) → epilogue(3,写 Y)；
    consumer_wait ∈ {"setup", "prefetch_w2", "mainloop", "none"}：在这个 phase 之前 wait（"none" = 不 wait）。
    """
    f = frozenset
    prod = Kernel("gemm1", (
        Phase("setup", 2.0),
        Phase("mainloop", 20.0, f({"X", "W1"})),
        Phase("epilogue", 4.0, writes=f({"H"})),
    ), trigger_after=1 if trigger_after_main else None)
    cons_phases = (
        Phase("setup", 2.0),
        Phase("prefetch_w2", 5.0, f({"W2"})),
        Phase("mainloop", 15.0, f({"H", "W2"})),
        Phase("epilogue", 3.0, writes=f({"Y"})),
    )
    names = [p.name for p in cons_phases]
    wait = None if consumer_wait == "none" else names.index(consumer_wait)
    cons = Kernel("gemm2", cons_phases, wait_before=wait)
    return [prod, cons]


def kernel_time(k: Kernel) -> float:
    """所有 phase 时长之和。"""
    return float(sum(p.us for p in k.phases))


def trigger_time(k: Kernel) -> float:
    """从 kernel 开始到 launch_dependents 的时间；没有 trigger 时等于 kernel_time。"""
    if k.trigger_after is None:
        return kernel_time(k)
    # trigger 在第 trigger_after 个 phase 结束之后发出
    return float(sum(p.us for p in k.phases[: k.trigger_after + 1]))


def timeline(kernels, mode: str, contention: float = 0.0) -> dict:
    """见模块 docstring。"""
    if mode not in ("stream", "graph", "pdl", "graph_pdl"):
        raise ValueError(f"unknown mode {mode!r}")
    use_pdl = mode in ("pdl", "graph_pdl")
    gap = GRAPH_GAP_US if mode in ("graph", "graph_pdl") else STREAM_GAP_US
    spans, phases = [], []
    prev = None
    prev_start = prev_end = 0.0
    for i, k in enumerate(kernels):
        if i == 0:
            start = 0.0
        elif use_pdl and prev.trigger_after is not None:
            # 后继 kernel 在前驱 trigger 之后启动；它的 phase 可以和前驱的尾巴重叠
            start = prev_start + trigger_time(prev) + PDL_LAUNCH_US
        else:
            # 没有 PDL（或前驱没有 trigger）：前驱完全结束后再加上空隙
            start = prev_end + gap
        if use_pdl and i > 0 and contention:
            # 先算“wait 之前”的部分占了多久，和前驱的名义结束时间比较，求重叠
            n_pre = len(k.phases) if k.wait_before is None else k.wait_before
            w = start + sum(p.us for p in k.phases[:n_pre])
            o = max(0.0, min(prev_end, w) - start)
            prev_end = prev_end + contention * o
            spans[-1] = (spans[-1][0], prev_end)
        t = start
        ph = []
        for j, p in enumerate(k.phases):
            if use_pdl and i > 0 and k.wait_before == j:
                # griddepcontrol.wait：等前驱完全结束并刷出内存，不是等到它的 trigger
                t = max(t, prev_end + FLUSH_US)
            ph.append((p.name, t, t + p.us))
            t += p.us
        if use_pdl and i > 0 and k.wait_before == len(k.phases):
            t = max(t, prev_end + FLUSH_US)
        spans.append((start, t))
        phases.append(ph)
        prev, prev_start, prev_end = k, start, t
    return {"spans": spans, "phases": phases, "total": max(e for _, e in spans)}


def best_trigger(kernels, mode: str = "pdl", contention: float = 0.0):
    """枚举 kernels[0] 的 trigger 位置，返回 (trigger_after, total)；并列时取更晚的 trigger。"""
    best = None
    for ta in range(len(kernels[0].phases)):
        ks = [replace(kernels[0], trigger_after=ta)] + list(kernels[1:])
        tot = timeline(ks, mode, contention)["total"]
        # 更晚的 trigger 在并列时优先：同样的收益下，少和前驱抢资源
        if best is None or tot <= best[1] + 1e-12:
            best = (ta, tot)
    return best


def _conflicts(prod: Kernel, phase: Phase):
    """phase 和 prod 冲突的 (kind, buf) 列表。"""
    p_writes = frozenset().union(*(p.writes for p in prod.phases))
    p_reads = frozenset().union(*(p.reads for p in prod.phases))
    out = []
    for b in sorted(phase.reads & p_writes):
        out.append(("read", b))                      # RAW：读前驱的输出
    for b in sorted(phase.writes & (p_writes | p_reads)):
        out.append(("write", b))                     # WAR / WAW：覆盖前驱还在读写的 buffer
    return out


def check(kernels) -> list:
    """见模块 docstring。"""
    issues = []
    for i in range(1, len(kernels)):
        prod, cons = kernels[i - 1], kernels[i]
        # 没有 wait 时，整个 kernel 都“在 wait 之前”
        limit = len(cons.phases) if cons.wait_before is None else cons.wait_before
        for j in range(limit):
            p = cons.phases[j]
            for kind, b in _conflicts(prod, p):
                issues.append(f"{kind}-before-wait: {cons.name}.{p.name} {kind}s {b}")
    return issues


def independent_operands(producer: Kernel, consumer: Kernel) -> list:
    """consumer 读的 buffer 里 producer 不写的那些，排序后返回（它们可以在 wait 之前加载）。"""
    p_writes = frozenset().union(*(p.writes for p in producer.phases))
    c_reads = frozenset().union(*(p.reads for p in consumer.phases))
    return sorted(c_reads - p_writes)


def latest_wait(producer: Kernel, consumer: Kernel) -> int:
    """最晚、仍然正确的 wait 位置。"""
    for j, p in enumerate(consumer.phases):
        if _conflicts(producer, p):
            return j
    return len(consumer.phases)


def speedup(kernels, base: str, opt: str) -> float:
    """timeline(base) 总时间 / timeline(opt) 总时间。"""
    return timeline(kernels, base)["total"] / timeline(kernels, opt)["total"]
