"""A5 · 把五个映射拼成一个完整的 Tiled GEMM：逐 CTA、逐 K 片模拟，并统计访存与指令 — reference solution.

Track A 的前四题分别回答了文档里的五个空间映射问题。这一题把它们首尾相连，得到一个真正“只靠映射”算出来的
64×64 GEMM：
    全局坐标 --local_tile--> CTA tile --Copy TV--> SMEM（swizzle）--MMA TV--> 寄存器 fragment
    --mma.sync--> 累加器（寄存器）--C TV--> 写回全局 C
结构对照 examples/cute/tutorial/sgemm_sm80.cu；每个线程的 partition 结果由 oracle/a5_assemble_gemm.cpp 用真实
CuTe 生成（tests/golden/a5_assemble_gemm.json）。

配置（与 A1–A4 相同）
- A (64, 64)、B (64, 64) 都是 K 连续（B 以 N×K 存放，“TN”），C = A · Bᵀ；half 元素 2 字节。
- CTA tile (BM, BN, BK) = (32, 32, 16)，grid 2×2，每个 CTA 128 个线程（4 个 warp），K循环 4 次。
- Copy：线程 t 负责 tile 行 t // 4、K 列 4·(t % 4) .. 4·(t % 4)+3（ThrLayout (32,4):(4,1)，ValLayout (1,4)）。
- SMEM：A、B 的 (32, 16) tile 各一块；tile 内坐标 (r, c) 的元素偏移 = smem_offset(r, c,swizzle)：
  swizzle=True 时为 Swizzle<2,2,2>(16·r + c)，False 时为 16·r + c，也可以传 (B, M, S)选别的 swizzle。
- MMA：A4 的 tiled_tv（given），每个 warp 每个 K 片执行 2 条 mma.sync。

统计口径（stats[(bm, bn)]，每个 CTA 一份）
- gmem_sectors：所有 copy 指令（每个 warp 每个 K 片对 A、B 各一条，每 lane 一次搬 4 个 half）触及的不同
  32 字节 sector 数之和。全局字节地址 = 2·(行 · 64 + 列)。
- smem_store_wavefronts：上述 copy 指令写 SMEM 的 wavefront 数之和。
- smem_load_wavefronts：smem → 寄存器的读取按“32 位 LDS”模型：每个 warp、每个 K 片，A 的4 个寄存器、
  B 的 4 个寄存器各一条指令（B 的寄存器 j 对应值 2j、2j+1），每 lane 读 4 个字节。
- wavefronts(byte_ranges)：一条指令里所有 lane 访问的字节区间 → 4 字节 word 的集合；
  wavefront 数 = max(每个 bank（word % 32）里不同 word 的最大个数, ceil(word 数 · 4 / 128))。同一个 word 的多次访问只算一次（广播）。
- mma：mma.sync 条数（= 4 warp × 2 × 4 K 片 = 32）；regs：每线程 32 位寄存器 {"A": 4, "B": 4, "C": 8}。

API
    local_tile_coords(bm, bn, kt, which) -> list[(r, c)]   given：A/B 的 (32,16) tile 或 C 的 (32,32) tile，
                                                            列主序（行变化最快）排列的全局坐标
    copy_coords(t) -> list[(r, c)]                          given：线程 t 在 (32,16)tile 里搬的 4 个 tile 内坐标
    smem_offset(r, c, swizzle=True) -> int                  given
    ptx_a, ptx_b, ptx_c, tiled_tv                           given：A4 的结果
    wavefronts(byte_ranges) -> int                          byte_ranges: list[(start_byte, nbytes)]
    simulate_gemm(A, B, swizzle=True) -> (C, stats)         stats = {(bm, bn): {...}, "c_writes": numpy 数组}
    trace(m, n, k) -> dict                                  追踪 C[m][n] 和它用到的 A[m][k]、B[n][k] 经过的每一层：
        {"cta": (bm, bn), "k_tile": kt,
         "A": {"tile": (r, c), "copy": (t, v), "smem": 偏移, "mma": [(t, v), ...]},
         "B": {"tile": (r, c), "copy": (t, v), "smem": 偏移, "mma": [(t, v), ...]},
         "C": {"tile": (r, c), "acc": (t, v)}}
"""
from __future__ import annotations

import numpy as np

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


def local_tile_coords(bm: int, bn: int, kt: int, which: str):
    """Global coordinates of CTA (bm, bn)'s tile (k-tile kt for A/B), column-major order."""
    if which == "A":
        return [(32 * bm + i, 16 * kt + j) for j in range(16) for i in range(32)]
    if which == "B":
        return [(32 * bn + i, 16 * kt + j) for j in range(16) for i in range(32)]
    if which == "C":
        return [(32 * bm + i, 32 * bn + j) for j in range(32) for i in range(32)]
    raise ValueError(which)


def copy_coords(t: int):
    """Copy TV of ThrLayout (32,4):(4,1), ValLayout (1,4): thread t copies 4 contiguous K of one row."""
    return [(t // 4, 4 * (t % 4) + v) for v in range(4)]


def smem_offset(r: int, c: int, swizzle=True) -> int:
    """Element offset in the (32,16) K-major SMEM tile.
    swizzle: True -> Swizzle<2,2,2> (the E3 choice), False -> plain (32,16):(16,1),
    or a tuple (B, M, S) -> Swizzle<B,M,S> o (32,16):(16,1)."""
    x = 16 * r + c
    if swizzle is True:
        swizzle = (2, 2, 2)
    if swizzle:
        B, M, S = swizzle
        x ^= (x & (((1 << B) - 1) << (M + S))) >> S
    return x


def ptx_a(lane: int, i: int):
    g, q = lane >> 2, lane & 3
    return g + 8 * ((i >> 1) & 1), 2 * q + (i & 1) + 8 * (i >> 2)


def ptx_b(lane: int, i: int):
    g, q = lane >> 2, lane & 3
    return g, 2 * q + (i & 1) + 8 * (i >> 1)


def ptx_c(lane: int, i: int):
    g, q = lane >> 2, lane & 3
    return g + 8 * (i >> 1), 2 * q + (i & 1)


def tiled_tv(which: str):
    """A4: TV map of the 2x2x1 TiledMMA with Tile<32,32,16>; offset = r + 32 c."""
    out = [0] * (128 * 8)
    for t in range(128):
        lane, wm, wn = t % 32, (t // 32) % 2, t // 64
        for v in range(8):
            if which == "A":
                r, c = ptx_a(lane, v)
                r += 16 * wm
            elif which == "B":
                r, c = ptx_b(lane, v % 4)
                r += 8 * (wn + 2 * (v // 4))
            else:
                r, c = ptx_c(lane, v % 4)
                r, c = r + 16 * wm, c + 8 * (wn + 2 * (v // 4))
            out[t + 128 * v] = r + 32 * c
    return out


def wavefronts(byte_ranges) -> int:
    """Shared-memory wavefronts of one warp instruction (32 banks x 4 bytes, broadcast free)."""
    words = set()
    for start, n in byte_ranges:
        for b in range(start, start + n):
            words.add(b // 4)
    if not words:
        return 0
    per_bank = {}
    for w in words:
        per_bank[w % 32] = per_bank.get(w % 32, 0) + 1
    # 两个下限取大：同一 bank 的不同 word 必须串行；一个 wavefront 最多服务 128 字节。
    return max(max(per_bank.values()), -(-len(words) * 4 // 128))


def _tv_coord(tv, t, v):
    off = tv[t + 128 * v]
    return off % 32, off // 32


def simulate_gemm(A, B, swizzle: bool = True):
    """C = A · Bᵀ computed purely through the five mappings; returns (C, stats)."""
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    if A.shape != (64, 64) or B.shape != (64, 64):
        raise ValueError("this drill fixes A and B to 64 x 64")
    tvA, tvB, tvC = tiled_tv("A"), tiled_tv("B"), tiled_tv("C")
    C = np.zeros((64, 64))
    c_writes = np.zeros((64, 64), dtype=np.int64)
    stats = {}
    for bm in range(2):
        for bn in range(2):
            st = {"gmem_sectors": 0, "smem_store_wavefronts": 0, "smem_load_wavefronts": 0,
                  "mma": 0, "regs": {"A": 4, "B": 4, "C": 8}}
            acc = np.zeros((128, 8))                       # 累加器留在寄存器里，跨越整个 K 循环
            for kt in range(4):
                sA, sB = np.full(512, np.nan), np.full(512, np.nan)
                # ---- gmem -> smem：每个 warp 一条 copy 指令（A、B 各一条）
                for X, S, which, row0 in ((A, sA, "A", 32 * bm), (B, sB, "B", 32 *bn)):
                    for w in range(4):
                        sectors, ranges = set(), []
                        for t in range(32 * w, 32 * w + 32):
                            cs = copy_coords(t)
                            for (r, c) in cs:
                                gr, gc = row0 + r, 16 * kt + c
                                sectors.add((2 * (gr * 64 + gc)) // 32)
                                S[smem_offset(r, c, swizzle)] = X[gr, gc]
                            # 一个 lane 的 4 个值：在 SMEM 里是否连续取决于 swizzle 的粒度（这里 4 个 half 一组，仍连续）
                            ranges.append((2 * smem_offset(*cs[0], swizzle), 8))
                        st["gmem_sectors"] += len(sectors)
                        st["smem_store_wavefronts"] += wavefronts(ranges)
                # ---- smem -> 寄存器：32 位 LDS 模型，每条指令每 lane 读 2 个相邻 half
                rA = np.zeros((128, 8))
                rB = np.zeros((128, 8))
                for w in range(4):
                    lanes = range(32 * w, 32 * w + 32)
                    for j in range(4):
                        for tv, S, R in ((tvA, sA, rA), (tvB, sB, rB)):
                            ranges = []
                            for t in lanes:
                                for v in (2 * j, 2 * j + 1):
                                    o = smem_offset(*_tv_coord(tv, t, v), swizzle)
                                    R[t, v] = S[o]
                                ranges.append((2 * smem_offset(*_tv_coord(tv, t, 2* j), swizzle), 4))
                            st["smem_load_wavefronts"] += wavefronts(ranges)
                # ---- 每个 warp 2 条 mma.sync（N 方向重复一次）
                for w in range(4):
                    for rep in range(2):
                        a, b = np.zeros((16, 16)), np.zeros((8, 16))
                        for lane in range(32):
                            t = 32 * w + lane
                            for i in range(8):
                                a[ptx_a(lane, i)] = rA[t, i]
                            for i in range(4):
                                b[ptx_b(lane, i)] = rB[t, i + 4 * rep]
                        cc = a @ b.T
                        for lane in range(32):
                            for i in range(4):
                                acc[32 * w + lane, i + 4 * rep] += cc[ptx_c(lane,i)]
                        st["mma"] += 1
            # ---- epilogue：K 全部完成后，按 C 的 TV 写回一次
            for t in range(128):
                for v in range(8):
                    r, c = _tv_coord(tvC, t, v)
                    C[32 * bm + r, 32 * bn + c] = acc[t, v]
                    c_writes[32 * bm + r, 32 * bn + c] += 1
            stats[(bm, bn)] = st
    stats["c_writes"] = c_writes
    return C, stats


def _holders(tv, r, c):
    off = r + 32 * c
    return [(i % 128, i // 128) for i, o in enumerate(tv) if o == off]


def trace(m: int, n: int, k: int) -> dict:
    """Follow C[m][n] and the A[m][k], B[n][k] it consumes through every layer."""
    if not (0 <= m < 64 and 0 <= n < 64 and 0 <= k < 64):
        raise ValueError("coordinates out of range")
    bm, bn, kt = m // 32, n // 32, k // 16
    out = {"cta": (bm, bn), "k_tile": kt}
    for name, row, tv in (("A", m - 32 * bm, tiled_tv("A")), ("B", n - 32 * bn, tiled_tv("B"))):
        col = k - 16 * kt
        t_copy = 4 * row + col // 4                    # copy 的逆：哪一个线程搬了这一行这 4 个 K
        out[name] = {"tile": (row, col), "copy": (t_copy, col % 4),
                     "smem": smem_offset(row, col), "mma": _holders(tv, row, col)}
    r, c = m - 32 * bm, n - 32 * bn
    out["C"] = {"tile": (r, c), "acc": _holders(tiled_tv("C"), r, c)[0]}
    return out
