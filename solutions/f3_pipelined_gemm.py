"""F3 · 多级 cp.async 流水线 + ldmatrix + 寄存器双缓冲：sgemm_sm80 的完整结构 — reference solution.

GPU 部分在 `f3_pipelined_gemm.cu`。主循环里最容易写错的是“哪一级 SMEM 在什么时候被写、什么时候被读”。
本文件把 .cu 里的循环（逐行照抄 CUTLASS sgemm_sm80.cu 的 L132–L305）用整数模拟一遍，得到完整的事件序列，
再检查它的不变量：每个 k-tile 恰好被算一次、按顺序、算的时候那一级 SMEM 里确实是它，而且已经 wait 过。

.cu 里的循环（S = STAGES，KB = BK / 16 = 4 个 k16 步）
    prefetch:  for s in 0..S-2:  load(k_next → slot s); commit; count -= 1; if count > 0: k_next += 1
    read = 0, write = S-1
    wait(S-2); sync                                   # 第 0 级到齐
    while count > -(S-1):                             # 一共转 k_tiles 圈
        for kb in 0..KB-1:
            if kb == KB-1: wait(S-2); sync            # 切到下一级之前，确认它到齐
            （读下一个 k16 的 fragment）
            if kb == 0:
                load(k_next → slot write); commit; count -= 1; if count > 0: k_next += 1
                write = read; read = (read + 1) % S
            compute(这一圈的 k-tile, kb)
- 一开始 count = k_tiles。k_next 在 count 降到 0 以后不再前进，所以最后 S-1 次 load 重复加载最后一个 k-tile
  （多余但无害：它写进的 slot 不会再被读）。
- 第 i 圈（i = 0..k_tiles-1）算的是第 i 个 k-tile，它在 slot i % S 里。
  注意：第 i 圈的 kb = 0 那一步先发了下一个 load 并把 read 往前推，但这一圈的 kb = 0..KB-1 用的
  fragment 是在上一圈末尾（或 prologue）从“旧的 read”那一级读出来的——这就是寄存器双缓冲。

trace(k_tiles, stages) 返回事件列表，每个事件是一个 tuple：
    ("load", k_tile, slot)       发出一次 cp.async（一个 commit group）
    ("wait", pending_allowed)    cp_async_wait<pending_allowed>() + __syncthreads()
    ("compute", k_tile, slot)    这一圈的第一个 k16 步（每圈只记一次）
规则：wait(n) 之后，除了最近的 n 个 group，之前发出的 group 都已经落地。

check_trace(events, stages)：重放事件，检查
    1. compute 的 k_tile 是 0, 1, 2, … 依次出现，每个一次；
    2. compute 时 slot 里最后一次 load 的就是这个 k_tile，并且那次 load 已经被某个 wait 确认落地；
    3. 一次 load 写进的 slot，不能是“已加载、尚未被 compute 用过”的那个 k_tile 所在的 slot（否则覆盖了没用过的数据）。
   全部满足返回 True，否则抛 AssertionError（消息写明哪一条）。

其他
    smem_bytes(BM, BN, BK, stages) = stages·(BM + BN)·BK·2
    max_stages(BM, BN, BK, limit=99·1024) = 放得下的最大 stage 数（至少 1，否则 ValueError）
    wait_arg(stages) = stages - 2                 cp_async_wait 的参数：允许还在飞的 group 数
    inflight_bytes(BM, BN, BK, stages) = (stages - 1)·(BM + BN)·BK·2   稳态时同时在路上的字节数
    redundant_loads(k_tiles, stages) = 总 load 次数 − k_tiles

API
    trace(k_tiles, stages) -> list[tuple]
    check_trace(events, stages) -> bool
    smem_bytes, max_stages, wait_arg, inflight_bytes, redundant_loads
"""
from __future__ import annotations


def trace(k_tiles: int, stages: int) -> list:
    if k_tiles < 1 or stages < 2:
        raise ValueError("need k_tiles >= 1 and stages >= 2")
    ev = []
    count, k_next = k_tiles, 0
    for s in range(stages - 1):
        ev.append(("load", k_next, s))
        count -= 1
        if count > 0:
            k_next += 1
    read, write = 0, stages - 1
    ev.append(("wait", stages - 2))
    i = 0
    while count > -(stages - 1):
        compute_slot = read            # 这一圈用的 fragment 来自 kb = 0 之前的 read
        # kb == 0：先发下一级 load，再推进 read / write
        ev.append(("load", k_next, write))
        count -= 1
        if count > 0:
            k_next += 1
        write, read = read, (read + 1) % stages
        ev.append(("compute", i, compute_slot))
        # kb == KB-1：切到下一级之前等它到齐
        ev.append(("wait", stages - 2))
        i += 1
    return ev


def check_trace(events: list, stages: int) -> bool:
    slot_tile = {}        # slot -> 最后一次 load 进来的 k_tile
    slot_group = {}       # slot -> 那次 load 的 group 编号
    groups = 0
    landed = -1           # 编号 <= landed 的 group 都已落地
    next_compute = 0
    loaded_unused = set()
    for e in events:
        if e[0] in ("load", "compute") and not 0 <= e[2] < stages:
            raise AssertionError(f"slot {e[2]} outside [0, {stages})")
        if e[0] == "load":
            _, kt, slot = e
            if slot in slot_tile and slot_tile[slot] in loaded_unused:
                raise AssertionError(f"rule 3: load of k-tile {kt} overwrites slot {slot} holding unused k-tile {slot_tile[slot]}")
            slot_tile[slot], slot_group[slot] = kt, groups
            groups += 1
            if kt >= next_compute:
                loaded_unused.add(kt)
        elif e[0] == "wait":
            landed = max(landed, groups - 1 - e[1])
        elif e[0] == "compute":
            _, kt, slot = e
            if kt != next_compute:
                raise AssertionError(f"rule 1: computed k-tile {kt}, expected {next_compute}")
            if slot_tile.get(slot) != kt:
                raise AssertionError(f"rule 2: slot {slot} holds {slot_tile.get(slot)}, not k-tile {kt}")
            if slot_group[slot] > landed:
                raise AssertionError(f"rule 2: k-tile {kt} computed before its cp.async group was waited for")
            loaded_unused.discard(kt)
            next_compute += 1
    return True


def smem_bytes(BM: int, BN: int, BK: int, stages: int) -> int:
    return stages * (BM + BN) * BK * 2


def max_stages(BM: int, BN: int, BK: int, limit: int = 99 * 1024) -> int:
    s = limit // ((BM + BN) * BK * 2)
    if s < 1:
        raise ValueError(f"even one stage of {BM}x{BN}x{BK} does not fit in {limit} bytes")
    return s


def wait_arg(stages: int) -> int:
    return stages - 2


def inflight_bytes(BM: int, BN: int, BK: int, stages: int) -> int:
    return (stages - 1) * (BM + BN) * BK * 2


def redundant_loads(k_tiles: int, stages: int) -> int:
    return sum(1 for e in trace(k_tiles, stages) if e[0] == "load") - k_tiles
