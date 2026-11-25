"""A2 · 线程怎样搬数据：Copy TV 映射、sector 与向量宽度 — reference solution.

从零设计 Tiled GEMM 的第二个问题：**128 个线程怎样协作把一个 32×16 的 A tile 从 global memory 搬走？**
文档里的设计是 ThrLayout = (32,4):(4,1)、ValLayout = (1,4)：每个线程搬同一行上 4 个连续的half。
本题从这两个 layout 出发，不用 CuTe API，自己推出 Copy TV 映射 (thread, value) → (m, k)，再数两种连续性：
线程之间（一个 warp 碰到几个 32 字节 sector）和线程内部（一个线程能合并成多宽的向量访问）。
期望值来自 oracle/a2_copy_tv.cpp：真实 CuTe 的 make_tiled_copy(...).get_layoutS_TV()和
get_slice(t).partition_S(identity tensor)。

规则
- 线程布局 thr = (thr_shape, thr_stride)，thr_shape = (TM, TK)：线程坐标 (tm, tk) 的线程号是
  tm·thr_stride[0] + tk·thr_stride[1]。它必须是 [0, TM·TK) 上的双射，否则 ValueError。
  thread_coord(thr_shape, thr_stride, t) 返回线程 t 的坐标 (tm, tk)（即线程布局的逆）。
- 值布局 val_shape = (VM, VK)，按 colex（列主序）编号：第 v 个值的局部坐标是 (v % VM, v //VM)。
- tile 形状 = (TM·VM, TK·VK)。线程 t 的第 v 个值复制的 tile 坐标（“raked”：值在内、线程在外）：
      (m, k) = (tm·VM + v % VM,  tk·VK + v // VM)
- copy_coords(...) 返回 [(m,k)]，先按线程 t = 0..T−1，再按值 v = 0..V−1（和 oracle 的 *_coords 顺序一致）。
- tv_index_map(...) 返回长度 T·V 的列表：第 i = t + T·v 项是 (m, k) 在列主序 tile 里的线性下标 m + TILE_M·k
  （这就是 CuTe get_layoutS_TV() 这个 layout 的完整取值表）。
- global 地址：A 是行主序、leading dimension = ld 的 half 矩阵，tile 原点 (m0, k0)：
      byte_addr(m, k) = ((m0 + m)·ld + (k0 + k))·elem_bytes
- vector_elems(...)：在 w ∈ {8, 4, 2, 1}（且 w·elem_bytes ≤ 16、V % w == 0）里取最大的w，使得每个线程的
  每一组值 v ∈ [g·w, (g+1)·w) 的字节地址都是连续递增的（相差 elem_bytes），且组首地址按 w·elem_bytes 对齐。
- copy_stats(...) 返回 dict：
      vec_elems = w，vec_bytes = w·elem_bytes，instr_per_thread = V // w
      sectors_per_warp = [每个 warp 所有指令触碰的 32 字节 sector 数之和]（warp = 连续 32 个线程号）
        一条指令 g 里，线程覆盖字节 [addr, addr + vec_bytes)，sector = byte // 32，按 warp 去重计数
      total_sectors = sum(sectors_per_warp)
      efficiency = tile 的总字节数 / (total_sectors·32)
- rank_variants(variants) 接收 {name: (thr_shape, thr_stride, val_shape)}，按
  (instr_per_thread, total_sectors, name) 升序返回名字列表：先少发指令，再少碰 sector。

API
    is_tuple, flatten, fmt_tuple, fmt_layout, parse_tuple, parse_layout     given（tools/cute_given.py）
    thread_coord(thr_shape, thr_stride, t) -> (int, int)
    copy_coords(thr_shape, thr_stride, val_shape) -> list[(int, int)]
    tv_index_map(thr_shape, thr_stride, val_shape) -> list[int]
    byte_addr(m, k, m0=0, k0=0, ld=64, elem_bytes=2) -> int
    vector_elems(thr_shape, thr_stride, val_shape, m0=0, k0=0, ld=64, elem_bytes=2)-> int
    copy_stats(thr_shape, thr_stride, val_shape, m0=0, k0=0, ld=64, elem_bytes=2) -> dict
    rank_variants(variants, m0=0, k0=0, ld=64, elem_bytes=2) -> list[str]
"""
from __future__ import annotations


def is_tuple(x) -> bool:
    return isinstance(x, tuple)


def flatten(t) -> tuple:
    """(2, (3, 4)) -> (2, 3, 4);  5 -> (5,)."""
    if not is_tuple(t):
        return (t,)
    out = ()
    for e in t:
        out += flatten(e)
    return out


def fmt_tuple(t, static: bool = True) -> str:
    """Format an IntTuple the way cute::print does: (_2,(_3,_4)) for static ints."""
    if is_tuple(t):
        return "(" + ",".join(fmt_tuple(e, static) for e in t) + ")"
    return ("_" if static else "") + str(t)


def fmt_layout(shape, stride, static: bool = True) -> str:
    """cute::print of a Layout: "<shape>:<stride>"."""
    return fmt_tuple(shape, static) + ":" + fmt_tuple(stride, static)


def parse_tuple(s: str):
    """Inverse of fmt_tuple. Accepts '_4' or '4'. '(_2,(_3,_4))' -> (2, (3, 4))."""
    s = s.strip()
    pos = 0

    def parse():
        nonlocal pos
        if s[pos] == "(":
            pos += 1
            items = []
            while s[pos] != ")":
                items.append(parse())
                if s[pos] == ",":
                    pos += 1
            pos += 1
            return tuple(items)
        start = pos
        while pos < len(s) and s[pos] not in ",)":
            pos += 1
        tok = s[start:pos].lstrip("_")
        return int(tok)

    v = parse()
    if pos != len(s):
        raise ValueError(f"trailing characters in {s!r}")
    return v


def parse_layout(s: str):
    """'(_4,_8):(_8,_1)' -> ((4, 8), (8, 1)). A swizzled/composed layout is not accepted."""
    shape_s, stride_s = _split_top_colon(s.strip())
    return parse_tuple(shape_s), parse_tuple(stride_s)


def _split_top_colon(s: str):
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ":" and depth == 0:
            return s[:i], s[i + 1:]
    raise ValueError(f"no top-level ':' in {s!r}")


# ---- A2: your code ----

def thread_coord(thr_shape, thr_stride, t: int) -> tuple:
    """线程 t 在线程布局里的坐标 (tm, tk)；线程布局不是双射时 ValueError。"""
    TM, TK = thr_shape
    s0, s1 = thr_stride
    # 线程布局是“坐标 → 线程号”的函数；搬运时我们需要反过来“线程号 → 坐标”，所以要求它是双射
    # （CuTe 里 make_tiled_copy 用 right_inverse 做同一件事，B 组再用代数重写）。
    inv = {}
    for tk in range(TK):
        for tm in range(TM):
            inv.setdefault(tm * s0 + tk * s1, []).append((tm, tk))
    if sorted(inv) != list(range(TM * TK)) or any(len(v) != 1 for v in inv.values()):
        raise ValueError("thread layout is not a bijection onto [0, size)")
    if not 0 <= t < TM * TK:
        raise ValueError("thread id out of range")
    return inv[t][0]


def copy_coords(thr_shape, thr_stride, val_shape) -> list:
    """[(m, k)]：线程 t = 0..T-1，每个线程的值 v = 0..V-1。"""
    TM, TK = thr_shape
    VM, VK = val_shape
    out = []
    for t in range(TM * TK):
        tm, tk = thread_coord(thr_shape, thr_stride, t)
        for v in range(VM * VK):
            # raked：一个线程的 VM×VK 个值是一个小块，线程网格决定小块放在哪里。
            out.append((tm * VM + v % VM, tk * VK + v // VM))
    return out


def tv_index_map(thr_shape, thr_stride, val_shape) -> list:
    """第 i = t + T·v 项 = (m, k) 在列主序 tile 里的下标 m + TILE_M·k。"""
    T = thr_shape[0] * thr_shape[1]
    V = val_shape[0] * val_shape[1]
    tile_m = thr_shape[0] * val_shape[0]
    cs = copy_coords(thr_shape, thr_stride, val_shape)
    # copy_coords 是 t-major；TV layout 的 colex 顺序是“线程维最快”，所以要转置一下下标。
    return [cs[(i % T) * V + i // T][0] + tile_m * cs[(i % T) * V + i // T][1] for i in range(T * V)]


def byte_addr(m: int, k: int, m0: int = 0, k0: int = 0, ld: int = 64, elem_bytes: int = 2) -> int:
    """行主序 A 里 tile 元素 (m, k) 的字节地址。"""
    return ((m0 + m) * ld + (k0 + k)) * elem_bytes


def _per_thread_addrs(thr_shape, thr_stride, val_shape, m0, k0, ld, elem_bytes):
    V = val_shape[0] * val_shape[1]
    cs = copy_coords(thr_shape, thr_stride, val_shape)
    return [[byte_addr(m, k, m0, k0, ld, elem_bytes) for (m, k) in cs[t * V:(t + 1)* V]]
            for t in range(len(cs) // V)]


def vector_elems(thr_shape, thr_stride, val_shape, m0=0, k0=0, ld=64, elem_bytes=2)-> int:
    """每个线程一次能合并多少个值成一个向量访问。"""
    V = val_shape[0] * val_shape[1]
    addrs = _per_thread_addrs(thr_shape, thr_stride, val_shape, m0, k0, ld, elem_bytes)
    for w in (8, 4, 2, 1):
        if w * elem_bytes > 16 or V % w:
            continue
        ok = True
        for a in addrs:
            for g in range(V // w):
                grp = a[g * w:(g + 1) * w]
                # 线程内部的连续性：w 个值地址连续、且组首按向量宽度对齐，才可能发一条 w·eb 字节的访问。
                if grp[0] % (w * elem_bytes) or any(grp[i + 1] - grp[i] != elem_bytes for i in range(w - 1)):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return w
    return 1


def copy_stats(thr_shape, thr_stride, val_shape, m0=0, k0=0, ld=64, elem_bytes=2) -> dict:
    """指令数、每个 warp 的 sector 数与效率。"""
    V = val_shape[0] * val_shape[1]
    w = vector_elems(thr_shape, thr_stride, val_shape, m0, k0, ld, elem_bytes)
    vb = w * elem_bytes
    addrs = _per_thread_addrs(thr_shape, thr_stride, val_shape, m0, k0, ld, elem_bytes)
    per_warp = []
    for w0 in range(0, len(addrs), 32):
        total = 0
        for g in range(V // w):
            # 线程之间的连续性：一条指令里整个 warp 碰到的 32 字节 sector 越少，事务越少。
            secs = set()
            for a in addrs[w0:w0 + 32]:
                start = a[g * w]
                secs.update(range(start // 32, (start + vb - 1) // 32 + 1))
            total += len(secs)
        per_warp.append(total)
    tile_bytes = len(addrs) * V * elem_bytes
    total_sectors = sum(per_warp)
    return {"vec_elems": w, "vec_bytes": vb, "instr_per_thread": V // w,
            "sectors_per_warp": per_warp, "total_sectors": total_sectors,
            "efficiency": tile_bytes / (total_sectors * 32)}


def rank_variants(variants, m0=0, k0=0, ld=64, elem_bytes=2) -> list:
    """按 (指令数, 总 sector 数, 名字) 升序排列方案名。"""

    def key(name):
        s = copy_stats(*variants[name], m0=m0, k0=k0, ld=ld, elem_bytes=elem_bytes)
        return (s["instr_per_thread"], s["total_sectors"], name)
    return sorted(variants, key=key)
