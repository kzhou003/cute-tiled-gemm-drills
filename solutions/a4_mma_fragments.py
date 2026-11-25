"""A4 · MMA fragment：从 PTX 的 groupID 规则推出 m16n8k16 的 TV_A / TV_B / TV_C，再扩展到 2×2 个 warp — reference solution.

第四个映射问题：Tensor Core 要求的数据怎样进入线程寄存器？SM80 的 mma.sync.m16n8k16 由一个 warp 的 32 个
lane 协作完成，每个 lane 只在寄存器里持有 A、B、C 的几个元素。PTX ISA 用两个量描述“谁持有谁”：
    groupID           = lane >> 2        （0..7，同一组 4 个 lane）
    threadID_in_group = lane %  4        （0..3）
本题先只用这两个公式（第一性原理）写出 TV 映射，再和 CuTe 的 MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>
（include/cute/atom/mma_traits_sm80.hpp）以及 make_tiled_mma(..., Layout<(2,2,1)>, Tile<32,32,16>) 的
get_layoutA/B/C_TV 对拍（golden 由 oracle/a4_mma_fragments.cpp 用真实 CuTe 生成）。

PTX 的 fragment 规则（f16 输入、f32 累加，“TN”：A 是 M×K 行、B 以 N×K 存放）
- A（16×16，每 lane 8 个 half，寄存器 a0..a7）：
      row = groupID + 8·((i >> 1) & 1)，  col = 2·threadID_in_group + (i & 1) + 8·(i >> 2)
- B（K×N = 16×8，每 lane 4 个 half，b0..b3）；本题用 (n, k) 表示（B 以 N×K 存放）：
      n = groupID，  k = 2·threadID_in_group + (i & 1) + 8·(i >> 1)
- C（16×8，每 lane 4 个 float，c0..c3）：
      row = groupID + 8·(i >> 1)，  col = 2·threadID_in_group + (i & 1)

约定
- atom 内的“列主序偏移”：A 为 m + 16·k，B 为 n + 8·k，C 为 m + 16·n。
- TV 的一维下标 i = t + T·v（线程维变化最快，CuTe 的 colex 约定）；atom_tv(which) 返回长度32·V 的偏移列表。
- TiledMMA：AtomLayout (2,2,1) 把 4 个 warp 排成 2×2，Tile<32,32,16> 让每个 warp 在 N方向重复一次。
  线程 t：lane = t % 32，warp 在 M 方向的坐标 wm = (t // 32) % 2，在 N 方向的坐标 wn = t// 64。
  - A（tile 32×16，偏移 m + 32·k）：V = 8，值 v 就是 atom 的 a_v；(m, k) = (am + 16·wm,ak)。
  - B（tile 32×16 的 (n, k)，偏移 n + 32·k）：V = 8，v = fv + 4·r（fv ∈ 0..3, r ∈ 0..1）：
      (n, k) = (bn + 8·(wn + 2·r), bk)
  - C（tile 32×32，偏移 m + 32·n）：V = 8，v = fv + 4·r：(m, n) = (cm + 16·wm, cn + 8·(wn + 2·r))
  其中 (am, ak)、(bn, bk)、(cm, cn) 是 atom 内坐标。这与真实 CuTe 的 get_layoutX_TV 完全一致。
- 寄存器计数（每线程，32 位寄存器）：A 8 个 half = 4 个，B 8 个 half = 4 个，C 8 个 float =8 个。
- fragment_shape(which)：partition_fragment_X 的 shape，按 CuTe 的 (MMA_V, MMA_M, MMA_N 或 MMA_K) 排列：
  A ((2,2,2),1,1)，B ((2,2),2,1)，C ((2,2),1,2)。
- 五个映射的依赖顺序（文档第 2 节）：MAPPINGS 按“先 CTA tile，再 copy，再 SMEM，再 MMA 输入，最后累加器”排列；
  DEPENDS[x] 列出 x 依赖的映射。valid_order(order) 判断一个顺序是否满足所有依赖（每个映射出现一次）。

API
    MAPPINGS, DEPENDS                                       given
    group_id(lane), thread_in_group(lane)                   lane → (groupID, threadID_in_group)
    ptx_a(lane, i) -> (m, k) ; ptx_b(lane, i) -> (n, k) ; ptx_c(lane, i) -> (m, n)
    atom_tv(which) -> list[int]                             which ∈ {"A","B","C"}
    tiled_tv(which) -> list[int]                            128 个线程的 TV 偏移（见上）
    registers_per_thread() -> dict                          {"A": 4, "B": 4, "C":8}
    fragment_shape(which) -> IntTuple
    holders(which, r, c) -> list[(t, v)]                    tiled 映射里持有 tile 元素 (r, c) 的所有 (t, v)
    warp_mma(a_frag, b_frag, c_frag) -> c_frag              一条 mma.sync：输入每 lane 的片段 (32,8)/(32,4)/(32,4)
    cta_tile_gemm(A, B) -> (C_tile, stats)                  A、B 是 (32, K) 的 numpy数组（B 以 N×K 存放），
                                                            K 是 16 的倍数；累加器一直留在 (128, 8) 的“寄存器”里；
                                                            stats = {"mma": mma.sync 条数, "k_slices": K/16,
                                                                     "acc_stores":写回 C 的次数（= 1024）}
    valid_order(order) -> bool
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


MAPPINGS = ["cta_tile", "copy_tv", "smem_layout", "mma_operand_tv", "accumulator_tv"]
DEPENDS = {
    "cta_tile": [],
    "copy_tv": ["cta_tile"],                       # 先知道 CTA 算哪块，才知道要搬什么
    "smem_layout": ["copy_tv"],                    # copy 的目的地是 SMEM 地址
    "mma_operand_tv": ["smem_layout"],             # SMEM 里有什么，才能谈怎样喂给 MMA
    "accumulator_tv": ["mma_operand_tv"],          # 累加器的分布由 MMA 指令决定
}


def group_id(lane: int) -> int:
    """PTX groupID = lane >> 2."""
    if not 0 <= lane < 32:
        raise ValueError("lane must be in [0, 32)")
    return lane >> 2


def thread_in_group(lane: int) -> int:
    """PTX threadID_in_group = lane % 4."""
    if not 0 <= lane < 32:
        raise ValueError("lane must be in [0, 32)")
    return lane & 3


def ptx_a(lane: int, i: int):
    """A fragment a_i of `lane`: (row m, col k) in the 16x16 A atom."""
    if not 0 <= i < 8:
        raise ValueError("A has 8 values per lane")
    g, q = group_id(lane), thread_in_group(lane)
    # 每个 32 位寄存器装 2 个相邻 K 的 half（i & 1）；a2/a3 是下半个 16×8 块（行 +8）；a4..a7 是 K 的后 8 列。
    return g + 8 * ((i >> 1) & 1), 2 * q + (i & 1) + 8 * (i >> 2)


def ptx_b(lane: int, i: int):
    """B fragment b_i of `lane`: (n, k) in the 8x16 B atom (B stored N x K)."""
    if not 0 <= i < 4:
        raise ValueError("B has 4 values per lane")
    g, q = group_id(lane), thread_in_group(lane)
    return g, 2 * q + (i & 1) + 8 * (i >> 1)


def ptx_c(lane: int, i: int):
    """C fragment c_i of `lane`: (m, n) in the 16x8 C atom."""
    if not 0 <= i < 4:
        raise ValueError("C has 4 values per lane")
    g, q = group_id(lane), thread_in_group(lane)
    return g + 8 * (i >> 1), 2 * q + (i & 1)


def atom_tv(which: str):
    """Atom TV map: offsets[t + 32 v] = column-major offset of value v of lanet."""
    if which == "A":
        V, f, rows = 8, ptx_a, 16
    elif which == "B":
        V, f, rows = 4, ptx_b, 8
    elif which == "C":
        V, f, rows = 4, ptx_c, 16
    else:
        raise ValueError(which)
    out = [0] * (32 * V)
    for v in range(V):
        for t in range(32):
            r, c = f(t, v)
            out[t + 32 * v] = r + rows * c          # CuTe 的 atom layout 输出的就是这个列主序偏移
    return out


def tiled_tv(which: str):
    """TV map of make_tiled_mma(atom, Layout<(2,2,1)>, Tile<32,32,16>) for A, B or C."""
    T, V = 128, 8
    out = [0] * (T * V)
    for t in range(T):
        lane, wm, wn = t % 32, (t // 32) % 2, t // 64
        for v in range(V):
            if which == "A":
                am, ak = ptx_a(lane, v)
                r, c = am + 16 * wm, ak                     # A 与 wn 无关：两个 N 方向的 warp 拿到同样的 A
            elif which == "B":
                fv, rep = v % 4, v // 4
                bn, bk = ptx_b(lane, fv)
                r, c = bn + 8 * (wn + 2 * rep), bk          # B 与 wm 无关；Tile<32>让每个 warp 在 N 上重复一次
            elif which == "C":
                fv, rep = v % 4, v // 4
                cm, cn = ptx_c(lane, fv)
                r, c = cm + 16 * wm, cn + 8 * (wn + 2 * rep)
            else:
                raise ValueError(which)
            out[t + T * v] = r + 32 * c
    return out


def registers_per_thread():
    """32-bit registers per thread for the A, B, C fragments of the tiled MMA."""
    # half 是 16 位：两个 half 装一个 32 位寄存器；float 累加器每个值一个寄存器。
    return {"A": 8 // 2, "B": 8 // 2, "C": 8}


def fragment_shape(which: str):
    """Shape of partition_fragment_X: (MMA_V, MMA_M, MMA_N or MMA_K)."""
    # 第一维是 atom 内每 lane 的值（按 PTX 寄存器的分组嵌套），后两维是这个线程在 tile 里重复了几次 atom。
    return {"A": ((2, 2, 2), 1, 1), "B": ((2, 2), 2, 1), "C": ((2, 2), 1, 2)}[which]


def holders(which: str, r: int, c: int):
    """All (t, v) whose tiled-TV entry is tile element (r, c)."""
    tv = tiled_tv(which)
    target = r + 32 * c
    return [(i % 128, i // 128) for i, off in enumerate(tv) if off == target]


def warp_mma(a_frag, b_frag, c_frag):
    """One mma.sync.m16n8k16: gather the lanes' fragments into A (16x16), B (8x16),add A·Bᵀ into C."""
    a_frag, b_frag, c_frag = np.asarray(a_frag), np.asarray(b_frag), np.asarray(c_frag)
    a, b = np.zeros((16, 16)), np.zeros((8, 16))
    for lane in range(32):
        for i in range(8):
            a[ptx_a(lane, i)] = a_frag[lane, i]
        for i in range(4):
            b[ptx_b(lane, i)] = b_frag[lane, i]
    c = a @ b.T
    out = c_frag.astype(np.float64).copy()
    for lane in range(32):
        for i in range(4):
            out[lane, i] += c[ptx_c(lane, i)]            # 每个 lane 只更新自己持有的4 个累加器
    return out


def cta_tile_gemm(A, B):
    """32x32 C tile = A · Bᵀ with the accumulator kept in per-thread registers across K slices."""
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    if A.shape[0] != 32 or B.shape[0] != 32 or A.shape[1] != B.shape[1] or A.shape[1] % 16:
        raise ValueError("A and B must be (32, K) with K a multiple of 16")
    tvA, tvB, tvC = tiled_tv("A"), tiled_tv("B"), tiled_tv("C")
    acc = np.zeros((128, 8))                              # 累加器：整个 K 循环都留在“寄存器”里
    n_mma = 0
    for ks in range(A.shape[1] // 16):
        a_t, b_t = A[:, 16 * ks:16 * ks + 16], B[:, 16 * ks:16 * ks + 16]
        rA = np.array([[a_t[tvA[t + 128 * v] % 32, tvA[t + 128 * v] // 32] for v in range(8)] for t in range(128)])
        rB = np.array([[b_t[tvB[t + 128 * v] % 32, tvB[t + 128 * v] // 32] for v in range(8)] for t in range(128)])
        for w in range(4):
            lanes = slice(32 * w, 32 * w + 32)
            for rep in range(2):                          # Tile<32> 在 N 方向的两次重复 = 两条 mma.sync
                c_part = acc[lanes, 4 * rep:4 * rep + 4]
                acc[lanes, 4 * rep:4 * rep + 4] = warp_mma(rA[lanes], rB[lanes, 4 *rep:4 * rep + 4], c_part)
                n_mma += 1
    C = np.zeros((32, 32))
    stores = 0
    for t in range(128):
        for v in range(8):
            off = tvC[t + 128 * v]
            C[off % 32, off // 32] = acc[t, v]            # K 全部完成后才写回一次
            stores += 1
    return C, {"mma": n_mma, "k_slices": A.shape[1] // 16, "acc_stores": stores}


def valid_order(order) -> bool:
    """True iff `order` lists every mapping once and each appears after everything it depends on."""
    if sorted(order) != sorted(MAPPINGS) or len(order) != len(MAPPINGS):
        return False
    seen = set()
    for x in order:
        if any(d not in seen for d in DEPENDS[x]):
            return False
        seen.add(x)
    return True
