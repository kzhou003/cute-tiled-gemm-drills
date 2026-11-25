"""A1 · CTA 切分：C tile、K 循环、local_tile 视图与边界谓词 — reference solution.

从零设计 Tiled GEMM 的第一个问题：**每个 CTA（Cooperative Thread Array，线程块）负责矩阵的哪一块？**
本题不用任何 CuTe API，只用整数运算把答案推出来；期望值来自 oracle/a1_cta_tiling.cpp：它对identity tensor（每个元素的值就是自己的坐标）调用真实 CuTe 的 local_tile，写出每个 CTA 视图覆盖的坐标。

运行例子（全部练习通用）
- A[M,K]、B[N,K]（两者都是 K 连续）、C[M,N] = A · Bᵀ；M = N = K = 64。
- CTA tile：BM = BN = 32，BK = 16 → 2×2 = 4 个 CTA，每个 CTA 的 K 循环 4 轮。
- 边界例子：M = 70、N = 50、K = 40（不是 tile 的整数倍），同样的 tile 大小。

规则
- num_tiles(extent, tile) = ceil(extent / tile)；extent、tile 必须为正整数，否则 ValueError。
- cta_grid(M, N, BM, BN) = (num_tiles(M, BM), num_tiles(N, BN))。
- 一个 CTA (pid_m, pid_n) 的 C tile 起点是 (pid_m·BM, pid_n·BN)。tile 一律按**完整的**BM×BN 计算，
  越界的部分用谓词屏蔽，而不是缩小 tile（这和 CuTe local_tile 的行为一致：视图形状总是静态的 tile 形状）。
- k_tile 号 k（0 ≤ k < num_tiles(K, BK)）覆盖 K 方向 [k·BK, (k+1)·BK)。
- 视图坐标顺序：一个 (R, S) 的 tile 视图，元素 i 是局部坐标 (i % R, i // R)，即 colex（第一维最快）。
  - c_tile_coords(pid_m, pid_n, BM, BN)          -> [(m, n), ...]   长度 BM·BN，m =pid_m·BM + i % BM，n = pid_n·BN + i // BM
  - a_tile_coords(pid_m, k, BM, BK)              -> [(m, kk), ...]  A 的 (BM, BK)视图：m 来自 pid_m，kk 来自 k_tile
  - b_tile_coords(pid_n, k, BN, BK)              -> [(n, kk), ...]  B 的 (BN, BK)视图：n 来自 pid_n（B 按 N×K 存）
  - 坐标可以越界（ragged 的最后一块）；越界由 predicate 判断。
- predicate(coords, shape) -> [bool, ...]：coord[d] < shape[d] 对每一维都成立才为 True（CuTe 的 elem_less）。
- view_offset(local, tile_origin, ld)：local_tile 只是**视图**——局部坐标 (r, c) 对应的
全局内存偏移是
  (origin_r + r)·ld + origin_c + c（行主序、leading dimension = ld），没有任何拷贝。
- verify_cover(M, N, BM, BN) -> bool：所有 CTA 的 C tile 里“谓词为真”的坐标恰好覆盖 [0,M)×[0,N) 每个元素一次。

API
    is_tuple, flatten, fmt_tuple, fmt_layout, parse_tuple, parse_layout     given（tools/cute_given.py）
    num_tiles(extent, tile) -> int
    cta_grid(M, N, BM, BN) -> (int, int)
    c_tile_coords(pid_m, pid_n, BM, BN) -> list[tuple[int, int]]
    a_tile_coords(pid_m, k, BM, BK) -> list[tuple[int, int]]
    b_tile_coords(pid_n, k, BN, BK) -> list[tuple[int, int]]
    predicate(coords, shape) -> list[bool]
    view_offset(local, tile_origin, ld) -> int
    verify_cover(M, N, BM, BN) -> bool
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


# ---- A1: your code ----

def num_tiles(extent: int, tile: int) -> int:
    """ceil(extent / tile)；extent、tile 必须是正整数，否则 ValueError。"""
    if extent <= 0 or tile <= 0:
        raise ValueError("extent and tile must be positive")
    # 向上取整：最后一块可能只有一部分在矩阵内，用谓词屏蔽。
    return (extent + tile - 1) // tile


def cta_grid(M: int, N: int, BM: int, BN: int) -> tuple:
    """(num_tiles(M, BM), num_tiles(N, BN))：CTA 网格的形状。"""
    return num_tiles(M, BM), num_tiles(N, BN)


def _tile_coords(r0: int, c0: int, R: int, S: int) -> list:
    # colex 顺序：第一维变化最快。CuTe 的 identity tensor 视图正是这个顺序（oracle 逐元素打印）。
    return [(r0 + i % R, c0 + i // R) for i in range(R * S)]


def c_tile_coords(pid_m: int, pid_n: int, BM: int, BN: int) -> list:
    """CTA (pid_m, pid_n) 的 C 视图覆盖的全局坐标，colex 顺序，长度 BM·BN。"""
    return _tile_coords(pid_m * BM, pid_n * BN, BM, BN)


def a_tile_coords(pid_m: int, k: int, BM: int, BK: int) -> list:
    """A 的 (BM, BK) 视图：行由 pid_m 决定，列由第 k 个 K tile 决定。"""
    # A 的视图只依赖 pid_m 和 k，不依赖 pid_n：同一行的两个 CTA 读同一块 A（数据复用发生在 L2）。
    return _tile_coords(pid_m * BM, k * BK, BM, BK)


def b_tile_coords(pid_n: int, k: int, BN: int, BK: int) -> list:
    """B 的 (BN, BK) 视图：B 按 N×K 存储，行由 pid_n 决定。"""
    return _tile_coords(pid_n * BN, k * BK, BN, BK)


def predicate(coords, shape) -> list:
    """每个坐标是否在 shape 之内（逐维 <）。"""
    return [all(c < s for c, s in zip(coord, shape)) for coord in coords]


def view_offset(local, tile_origin, ld: int) -> int:
    """视图里局部坐标 local=(r, c) 对应的全局行主序偏移：(origin_r + r)·ld + origin_c + c。"""
    r, c = local
    r0, c0 = tile_origin
    # local_tile 不搬数据：它只是把“局部坐标”平移成“全局坐标”，再用原 tensor 的 layout 求偏移。
    return (r0 + r) * ld + (c0 + c)


def verify_cover(M: int, N: int, BM: int, BN: int) -> bool:
    """有效（谓词为真）的 C 坐标恰好覆盖 [0,M)×[0,N) 每个元素一次。"""
    gm, gn = cta_grid(M, N, BM, BN)
    seen = {}
    for pm in range(gm):
        for pn in range(gn):
            cs = c_tile_coords(pm, pn, BM, BN)
            for c, ok in zip(cs, predicate(cs, (M, N))):
                if ok:
                    seen[c] = seen.get(c, 0) + 1
    return len(seen) == M * N and all(v == 1 for v in seen.values())
