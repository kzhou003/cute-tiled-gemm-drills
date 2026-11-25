"""A3 · Shared memory 怎样摆：行主序、Swizzle 与 bank conflict — reference solution.

从零设计 Tiled GEMM 的第三个问题：**A tile 搬进 shared memory（SMEM）以后，每个元素放在哪个地址？**
文档的运行例子用 SmemLayout = Swizzle<2,2,2> ∘ (32,16):(16,1)：先有普通的行主序地址映射，再复合一个只改
地址、不改元素身份的位运算。本题从零实现这个映射，再用一个 bank 模型实测：对三种真实的访问模式
（A2 的 8 字节拷贝写入、ldmatrix 式的 16 字节行读、MMA A 操作数的 4 字节读），哪种 swizzle 能消除冲突。
期望的地址表来自 oracle/a3_smem_swizzle.cpp：真实 CuTe 的 composition(Swizzle<B,M,S>{},layout)。

地址映射（单位：half 元素）
- plain：(m, k) → 16·m + k（32 行 × 16 列，每行 32 字节）。
- swizzle(x, B, M, S)（CuTe 的 Swizzle<B,M,S>）：B ≥ 0、M ≥ 0、|S| ≥ B，否则 ValueError。
      yyy = ((1<<B)-1) << (M + max(0, S))，  swizzle(x) = x ^ shiftr(x & yyy, S)
  shiftr(v, s) 在 s ≥ 0 时右移 s 位，s < 0 时左移 −s 位。
- smem_offset(m, k, sw=None)：sw 为 None 时是 plain，否则 swizzle(plain(m, k), *sw)。
- smem_map(sw=None)：长度 512 的列表，第 i 项是 tile 元素 (i % 32, i // 32) 的偏移（colex，与 oracle 一致）。
- is_bijection(values)：values 恰好是 0..len−1 的排列。
- chunk_contiguous(sw, w)：每一行里每个对齐的 w 元素组 k ∈ [j·w, (j+1)·w)，偏移按顺序连续递增。

bank 模型（简化模型）
- 32 个 bank，每个 4 字节：word = byte // 4，bank = word % 32。
- 每个线程访问 access_bytes ∈ {4, 8, 16} 字节，地址必须按 access_bytes 对齐（否则 ValueError）；
  线程按顺序每 128 // access_bytes 个分成一个 phase。phase 的 wavefront 数 = 各 bank 上“不同 word 的个数”的最大值
  （同一个 word 多个线程读是广播）。wavefronts(addrs, access_bytes) 返回所有 phase 的 wavefront 之和。

三种访问模式（地址都是字节，元素 2 字节）。所需的 w 个元素在 sw 下不连续时抛 ValueError（这个 swizzle 会拆散向量）。
- copy_store_addrs(sw, order="base")：A2 的拷贝写入，128 个线程各写 4 个 half（8 字节）。
      order="base"：线程 t 写 (t // 4, 4·(t % 4))；order="colthr"：线程 t 写 (t % 32,4·(t // 32))。
- ldmatrix_addrs(sw, r0, c0)：一个 8×8 矩阵，线程 i（0..7）给出 (r0 + i, c0) 起的 16 字节行地址。
- lds32_addrs(sw, r0=0, k0=0)：MMA A 操作数的 4 字节读，lane l（0..31）读 (r0 + l // 4,k0 + 2·(l % 4)) 起的 2 个 half。
- conflict_report(sw) -> dict（每项是 wavefront 总数，括号里是无冲突时的值）：
      "store_base"   = 4 个 warp 的 copy_store_addrs(sw, "base") 之和（理想 8）
      "store_colthr" = 同上，order="colthr"（理想 8）
      "ldmatrix"     = 8×8 块 (r0, c0) ∈ {0, 8, 16, 24} × {0, 8} 之和（理想 8）
      "lds32"        = (r0, k0) ∈ {0, 8, 16, 24} × {0, 8} 之和（理想 8）
  某个模式在 sw 下非法（ValueError）时该项为 None。
- find_min_swizzle(pattern, max_B=3, max_M=4, max_S=5)：按 B、M、S 从小到大（S 从 max(B,1) 到 max_S）枚举，跳过
  M + max(0,S) + B > 9（超出 512 元素的定义域）和使该模式非法的候选，返回第一个让 conflict_report[pattern]
  等于理想值 8 的 (B, M, S)；B = 0 视为不 swizzle，返回 (0, 0, 0)。都不行返回 None。

API
    is_tuple, flatten, fmt_tuple, fmt_layout, parse_tuple, parse_layout     given（tools/cute_given.py）
    swizzle(x, B, M, S) -> int
    smem_offset(m, k, sw=None) -> int
    smem_map(sw=None) -> list[int]
    is_bijection(values) -> bool
    chunk_contiguous(sw, w) -> bool
    wavefronts(byte_addrs, access_bytes) -> int
    copy_store_addrs(sw, order="base") -> list[int]
    ldmatrix_addrs(sw, r0, c0) -> list[int]
    lds32_addrs(sw, r0=0, k0=0) -> list[int]
    conflict_report(sw) -> dict
    find_min_swizzle(pattern, max_B=3, max_M=4, max_S=5) -> tuple | None
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
# ---- A3: your code ----

def swizzle(x: int, B: int, M: int, S: int) -> int:
    """CuTe 的 Swizzle<B,M,S>：把 yyy 位右移 S 位后异或到 zzz 位上。"""
    if B < 0 or M < 0 or abs(S) < B:
        raise ValueError("Swizzle needs B >= 0, M >= 0, |S| >= B")
    yyy = ((1 << B) - 1) << (M + max(0, S))
    moved = x & yyy
    # 只做位运算：同一个地址永远映射到同一个新地址，而且它是自己的逆（异或两次复原）。
    return x ^ (moved >> S if S >= 0 else moved << -S)


def smem_offset(m: int, k: int, sw=None) -> int:
    """逻辑坐标 (m, k) 在 SMEM 里的元素偏移。"""
    plain = 16 * m + k
    # 先算普通 layout，再套 swizzle：Swizzle ∘ Layout（CuTe ComposedLayout 的求值顺序）。
    return plain if sw is None else swizzle(plain, *sw)


def smem_map(sw=None) -> list:
    """colex 顺序（m 最快）的 512 个偏移。"""
    return [smem_offset(i % 32, i // 32, sw) for i in range(512)]


def is_bijection(values) -> bool:
    """values 是否恰好是 0..len-1 的一个排列。"""
    return sorted(values) == list(range(len(values)))


def chunk_contiguous(sw, w: int) -> bool:
    """每行里每个对齐的 w 元素组在 SMEM 里是否仍按顺序连续。"""
    for m in range(32):
        for j in range(16 // w):
            offs = [smem_offset(m, j * w + i, sw) for i in range(w)]
            if any(offs[i + 1] - offs[i] != 1 for i in range(w - 1)):
                return False
    return True


def wavefronts(byte_addrs, access_bytes: int) -> int:
    """所有 phase 的 wavefront 数之和。"""
    if access_bytes not in (4, 8, 16):
        raise ValueError("access_bytes must be 4, 8 or 16")
    if any(a % access_bytes for a in byte_addrs):
        raise ValueError("misaligned access")
    per_phase = 128 // access_bytes
    total = 0
    for p in range(0, len(byte_addrs), per_phase):
        banks = {}
        for a in byte_addrs[p:p + per_phase]:
            for wd in range(a // 4, (a + access_bytes) // 4):
                banks.setdefault(wd % 32, set()).add(wd)
        # 一个 bank 上有几个不同的 word，就要分几轮；广播（同一个 word）不额外计数。
        total += max(len(s) for s in banks.values())
    return total


def _vector(sw, m: int, k: int, w: int) -> int:
    offs = [smem_offset(m, k + i, sw) for i in range(w)]
    if any(offs[i + 1] - offs[i] != 1 for i in range(w - 1)):
        raise ValueError("this swizzle breaks a %d-element vector" % w)
    return 2 * offs[0]


def copy_store_addrs(sw, order: str = "base") -> list:
    """A2 拷贝的 128 个 8 字节写入地址。"""
    out = []
    for t in range(128):
        m, k = (t // 4, 4 * (t % 4)) if order == "base" else (t % 32, 4 * (t // 32))
        out.append(_vector(sw, m, k, 4))
    return out


def ldmatrix_addrs(sw, r0: int, c0: int) -> list:
    """8×8 矩阵的 8 个 16 字节行地址。"""
    return [_vector(sw, r0 + i, c0, 8) for i in range(8)]


def lds32_addrs(sw, r0: int = 0, k0: int = 0) -> list:
    """MMA A 操作数 32 个 lane 的 4 字节读地址。"""
    return [_vector(sw, r0 + l // 4, k0 + 2 * (l % 4), 2) for l in range(32)]


def _safe(fn):
    try:
        return fn()
    except ValueError:
        return None


def conflict_report(sw) -> dict:
    """四种访问模式的 wavefront 总数（非法为 None）。"""

    def stores(order):
        a = copy_store_addrs(sw, order)
        return sum(wavefronts(a[w:w + 32], 8) for w in range(0, 128, 32))
    blocks = [(r, c) for r in (0, 8, 16, 24) for c in (0, 8)]
    return {
        "store_base": _safe(lambda: stores("base")),
        "store_colthr": _safe(lambda: stores("colthr")),
        "ldmatrix": _safe(lambda: sum(wavefronts(ldmatrix_addrs(sw, r, c), 16) for r, c in blocks)),
        "lds32": _safe(lambda: sum(wavefronts(lds32_addrs(sw, r, c), 4) for r, c in blocks)),
    }


def find_min_swizzle(pattern: str, max_B: int = 3, max_M: int = 4, max_S: int = 5):
    """第一个让该访问模式无冲突的 (B, M, S)。"""
    if conflict_report(None)[pattern] == 8:
        return (0, 0, 0)
    for B in range(1, max_B + 1):
        for M in range(0, max_M + 1):
            for S in range(max(B, 1), max_S + 1):
                if M + max(0, S) + B > 9:
                    continue
                if conflict_report((B, M, S))[pattern] == 8:
                    return (B, M, S)
    return None
