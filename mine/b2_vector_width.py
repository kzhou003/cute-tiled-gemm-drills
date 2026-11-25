"""B2 · 一次能搬多宽：max_common_vector、swizzle 的上限、对齐与 copy atom 的选择 — your implementation (run ./check.sh b2).

A2 里每个线程一次搬 4 个 half（8 字节），这个“4”是我们拍脑袋定的。CuTe 不靠拍脑袋：make_tiled_copy 之前，
CUTLASS 用 max_common_vector(gmem 视图, smem layout) 求出“两边同时连续”的最长元素段，再挑一个不超过它的
copy atom（文档第 6 节“编译器能优化什么”：向量化是从 layout 推出来的，不是写死的）。这一题实现这个推导，
并看清它的边界：swizzle 把上限压到 2^M，padding 让 layout 上的答案和真实对齐不一致。
期望结果来自 oracle/b2_vector_width.cpp（真实 CuTe 的 max_common_vector / max_common_layout）。

约定
- 所有 layout 的定义域是 tile 坐标 (m, k)，按 colex 编号 i（m 最快），值是元素偏移（不是字节）。
- 普通 layout 用 given 的 Layout；swizzle 过的 SMEM layout 用 given 的 Swizzled(B, M,S, base)：
  Swizzled(i) = swizzle(B, M, S, base(i))，repr 和 CuTe 一致：Sw<2,2,2> o _0 o (_32,_16):(_16,_1)。
- half：elem_bytes = 2。

API（Layout、composition、coalesce、right_inverse、swizzle、Swizzled 等是 given）
    max_common_vector(a, b) -> int
        普通 × 普通：common = coalesce(composition(a, right_inverse(b)))；第 0 个模式stride 为 1 时返回它的 size，否则 1。
        任意一边是 Swizzled：先对 base 求，再和 2^M 取 min；两边都是 Swizzled：同一个 swizzle 时不截断，
        不同时和两个 2^M 取 min（swizzle_layout.hpp 的规则）。
    max_common_layout(a, b) -> Layout
        普通 × 普通：inv = right_inverse(b)；第 0 个模式 stride 为 1 时返回 composition(inv, 那个模式)，否则 _1:_0。
        结果的值是 tile 内的 colex 下标 i：这 n 个元素在两边都连续。Swizzled 一边：截断到前2^M 个（composition(common, Layout(2^M))）。
    brute_vector(a, b, size) -> int
        不用代数的定义：最大的 2 的幂 n（n 整除 size），使得对每个 b(i) % n == 0 的 i：a(i)% n == 0，
        并且 b 值为 b(i)+j（0 < j < n）的元素都存在、它在 a 里的值正好是 a(i)+j。
        （b 里每个对齐的 n 元素块，在 a 里也是对齐的连续块：这就是硬件向量访存的条件。）
    vector_bytes(n_elems, elem_bytes=2, cap=16) -> int
        不超过 cap、且整除 n_elems·elem_bytes 的最大 2 的幂字节数。例：16 个 half → 16；3个 half → 2。
    alignment_bytes(byte_offsets, cap=16) -> int    所有偏移的公共 2 的幂对齐（不超过 cap）；全 0 时返回 cap
    copy_atom(nbytes, async_=False) -> str
        同步：UniversalCopy<uint16_t | uint32_t | uint64_t | uint128_t>（2/4/8/16 字节）。
        cp.async：SM80_CP_ASYNC_CACHEALWAYS<uint32_t | uint64_t | uint128_t>（4/8/16 字节，cp.async.ca，
        sgemm_sm80.cu 用的就是 <uint128_t>）。其它大小 ValueError。
    max_alignment(layout) -> int    这个 layout 能保证的对齐（元素个数），layout.hpp 的 max_alignment：
        f = coalesce(layout)；p = logical_divide(f, right_inverse(f))；返回 gcd(size(p 的第 0 个模式), p 第 1 个模式的 stride)
        （stride 是元组时取展平后的第一个）。Swizzled：gcd(2^M, max_alignment(base))。
        例：(32,16):(18,1) → 2（行起点 18r 只保证 2 个元素对齐），(32,16):(16,1) → 512。
    auto_vector_bits(a, b, elem_bits=16, max_vec_bits=128) -> int
        algorithm/copy.hpp 里 AutoVectorizingCopyWithAssumedAlignment 的公式：
        mcv = max_common_vector(a, b)；mcv ≤ 1 时返回 elem_bits；
        align = gcd(max_alignment(a)·elem_bits, max_alignment(b)·elem_bits, max_vec_bits)；
        返回 gcd(mcv·elem_bits, align)。
    copy_plan(a, b, vals_per_thread, elem_bits=16, base_offsets_bytes=(0,), async_=False) -> dict
        {"mcv": max_common_vector(a, b),
         "bytes": min(auto_vector_bits(a, b) / 8, vector_bytes(vals_per_thread), alignment_bytes(base_offsets_bytes)),
         "vec": bytes / 元素字节数, "atom": copy_atom(bytes, async_)}
        base_offsets_bytes 是 tensor 起点相对 16 字节对齐地址的偏移（layout 看不到的那部分对齐）。
"""
from __future__ import annotations

from math import gcd

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
# ---------------------------------------------------------------- shared core (given) -----------
# Static-integer mirror of CuTe's Layout algebra at the pinned commit (include/cute/layout.hpp).
from math import prod as _mprod


class Underscore:
    """CuTe's `_` (also spelled X in Step<...>): keep this mode when slicing; drop it when dicing."""

    def __repr__(self):
        return "_"


_ = Underscore()
X = _


def is_underscore(x) -> bool:
    return isinstance(x, Underscore)


def product(t) -> int:
    """Product of all leaves of an IntTuple."""
    return _mprod(flatten(t))


def rank(t) -> int:
    return len(t) if is_tuple(t) else 1


def get(t, i):
    """get<i>(t); get<0> of a non-tuple is the value itself (CuTe convention)."""
    if is_tuple(t):
        return t[i]
    if i != 0:
        raise IndexError(i)
    return t


def unwrap(t):
    while is_tuple(t) and len(t) == 1:
        t = t[0]
    return t


def ceil_div(a, b):
    """cute::ceil_div on IntTuples (int_tuple.hpp)."""
    if is_tuple(a):
        if is_tuple(b):
            b = tuple(b) + (1,) * (len(a) - len(b))
            return tuple(ceil_div(x, y) for x, y in zip(a, b))
        out, rest = (), b
        for ai in a:
            out += (ceil_div(ai, rest),)
            rest = ceil_div(rest, ai)
        return out
    if is_tuple(b):
        return ceil_div(a, product(b))
    return (a + b - 1) // b


def crd2idx(coord, shape, stride):
    """stride.hpp crd2idx(c, s, d). An int coord on a tuple shape is split colexicographically and
    the LAST mode absorbs the remainder (so indices past size() extend the last mode).
    Underscore coordinates count as 0 (this is the offset used when slicing)."""
    if is_underscore(coord):
        return 0
    if is_tuple(coord):
        if not is_tuple(shape) or len(coord) != len(shape):
            raise ValueError("mismatched ranks")
        return sum(crd2idx(c, s, d) for c, s, d in zip(coord, shape, stride))
    if is_tuple(shape):
        total, c = 0, coord
        for k, (s, d) in enumerate(zip(shape, stride)):
            if k == len(shape) - 1:
                total += crd2idx(c, s, d)
            else:
                c, m = divmod(c, product(s))
                total += crd2idx(m, s, d)
        return total
    return coord * stride


def idx2crd(idx, shape, stride):
    """stride.hpp idx2crd(i, s, d): (i // d) % s per leaf (inverse of a compact layout)."""
    if is_tuple(shape):
        return tuple(idx2crd(idx, s, d) for s, d in zip(shape, stride))
    if shape == 1:          # size-1 mode: coordinate is 0 (skips a possible stride-0 division)
        return 0
    return (idx // stride) % shape


def crd2crd_1d(coord, shape):
    """Colexicographic 1-D index of a coordinate inside one mode's shape (crd2idx(c, s))."""
    if not is_tuple(coord):
        return coord
    fc, fs = flatten(coord), flatten(shape)
    i = 0
    for c, s in zip(reversed(fc), reversed(fs)):
        i = i * s + c
    return i


class Layout:
    """A CuTe Layout (Shape, Stride) of static ints. Calling it maps coord/index ->offset."""

    def __init__(self, shape, stride=None):
        if stride is None:
            stride = _compact_left(shape)
        self.shape, self.stride = shape, stride

    # -- function view
    def __call__(self, *c):
        c = c[0] if len(c) == 1 else tuple(c)
        if _has_underscore(c):
            raise TypeError("use slice_layout(coord, layout) for coordinates containing _")
        return crd2idx(c, self.shape, self.stride)

    def size(self) -> int:
        return product(self.shape)

    def cosize(self) -> int:
        return self(self.size() - 1) + 1 if self.size() > 0 else 0

    def rank(self) -> int:
        return rank(self.shape)

    def mode(self, i) -> "Layout":
        return Layout(get(self.shape, i), get(self.stride, i))

    def map(self) -> list:
        """[L(0), L(1), ..., L(size-1)]"""
        return [self(i) for i in range(self.size())]

    def __eq__(self, other):
        return isinstance(other, Layout) and (self.shape, self.stride) == (other.shape, other.stride)

    def __hash__(self):
        return hash((self.shape, self.stride))

    def __repr__(self):
        return fmt_layout(self.shape, self.stride)


def _compact_left(shape, start=1):
    """compact_major<LayoutLeft>: column-major strides for shape, starting at `start`.
    A size-1 mode gets stride 0 (stride.hpp detail::compact), e.g. (4,1) -> (1,0)."""
    if is_tuple(shape):
        out, cur = [], start
        for s in shape:
            out.append(_compact_left(s, cur))
            cur *= product(s)
        return tuple(out)
    return 0 if shape == 1 else start


def _has_underscore(c) -> bool:
    if is_tuple(c):
        return any(_has_underscore(e) for e in c)
    return is_underscore(c)


def make_layout(*layouts) -> Layout:
    """make_layout(L0, L1, ...): concatenation; ONE argument still wraps: shape (s0,)."""
    return Layout(tuple(l.shape for l in layouts), tuple(l.stride for l in layouts))


def _as_layout(t) -> Layout:
    """A tiler element that is an int n means the layout n:1 (make_layout(n))."""
    return t if isinstance(t, Layout) else Layout(t, 1 if not is_tuple(t) else _compact_left(t))


def _append_mode(shape, stride, s, d):
    if is_tuple(shape):
        return shape + (s,), stride + (d,)
    return (shape, s), (stride, d)


def _bw_coalesce(old_shape, old_stride, new_shape, new_stride) -> Layout:
    """layout.hpp detail::bw_coalesce, iterating from the back."""
    for s, d in zip(reversed(old_shape), reversed(old_stride)):
        if s == 1:
            continue
        if new_shape == 1:
            new_shape, new_stride = s, d
        elif s * d == get(new_stride, 0):
            if is_tuple(new_shape):
                new_shape = (s * new_shape[0],) + new_shape[1:]
                new_stride = (d,) + new_stride[1:]
            else:
                new_shape, new_stride = s * new_shape, d
        else:
            if is_tuple(new_shape):
                new_shape, new_stride = (s,) + new_shape, (d,) + new_stride
            else:
                new_shape, new_stride = (s, new_shape), (d, new_stride)
    if new_shape == 1:
        return Layout(1, 0)
    return Layout(new_shape, new_stride)


def coalesce(layout: Layout, profile=None) -> Layout:
    """cute::coalesce; with a tuple profile, coalesce each mode separately (transform_layout)."""
    if is_tuple(profile):
        return transform_layout(layout, profile, lambda l, t: coalesce(l, t))
    fs, fd = flatten(layout.shape), flatten(layout.stride)
    return _bw_coalesce(fs[:-1], fd[:-1], fs[-1], fd[-1])


def coalesce_x(layout: Layout) -> Layout:
    """detail::coalesce_x: like coalesce but keeps the function correct OUTSIDE the domain too:
    a terminal size-1 mode starts the fold as shape 2 so its stride survives."""
    fs, fd = flatten(layout.shape), flatten(layout.stride)
    last = 2 if fs[-1] == 1 else fs[-1]
    return _bw_coalesce(fs[:-1], fd[:-1], last, fd[-1])


def filter_layout(layout: Layout) -> Layout:
    """cute::filter: replace stride-0 modes by size 1, then coalesce."""

    def fz(s, d):
        if is_tuple(s):
            return tuple(fz(a, b) for a, b in zip(s, d))
        return 1 if d == 0 else s
    return coalesce(Layout(fz(layout.shape, layout.stride), layout.stride))


def transform_layout(t0: Layout, t1, f) -> Layout:
    """transform_layout(layout, tuple, f): f on the first min(rank) modes, remaining modes of
    BOTH appended unchanged (here: the leftover layout modes; leftover tiler modes are invalid)."""
    r0, r1 = t0.rank(), len(t1)
    if r1 > r0:
        raise ValueError("too many modes in tiler")
    modes = [f(t0.mode(i), t1[i]) for i in range(r1)] + [t0.mode(i) for i in range(r1, r0)]
    return make_layout(*modes)


def _composition_impl(lhs_shape, lhs_stride, rhs_shape, rhs_stride) -> Layout:
    if is_tuple(rhs_shape):                                  # right-distributivity
        return make_layout(*[_composition_impl(lhs_shape, lhs_stride, s, d)
                             for s, d in zip(rhs_shape, rhs_stride)])
    if rhs_stride == 0:                                      # static stride-0 shortcut
        return Layout(rhs_shape, rhs_stride)
    if not is_tuple(lhs_shape):                              # integral lhs
        return Layout(rhs_shape, rhs_stride * lhs_stride)
    R = len(lhs_shape)
    res_s, res_d, rest_s, rest_d = (), (), rhs_shape, rhs_stride
    for i in range(R - 1):
        cs, cd = lhs_shape[i], lhs_stride[i]
        if not (abs(rest_d) % cs == 0 or abs(rest_d) < cs):
            raise ValueError("Stride Divisibility Condition")
        next_s = ceil_div(cs, abs(rest_d))
        next_d = ceil_div(abs(rest_d), cs) * (1 if rest_d >= 0 else -1)
        if next_s == 1 or rest_s == 1:
            rest_d = next_d
            continue
        new_s = min(next_s, rest_s)
        if rest_s % new_s != 0:
            raise ValueError("Shape Divisibility Condition")
        res_s, res_d = res_s + (new_s,), res_d + (rest_d * cd,)
        rest_s, rest_d = rest_s // new_s, next_d
    if len(res_s) == 0:
        return Layout(rest_s, rest_d * lhs_stride[R - 1])
    if rest_s == 1:
        return Layout(unwrap(res_s), unwrap(res_d))
    return Layout(res_s + (rest_s,), res_d + (rest_d * lhs_stride[R - 1],))


def composition(lhs: Layout, rhs) -> Layout:
    """cute::composition(lhs, rhs). rhs: Layout, int n (== n:1), _ (identity) or a tuple tiler
    (by-mode; lhs modes NOT hit by the tiler are DROPPED, unlike logical_divide)."""
    if isinstance(rhs, Layout):
        f = coalesce_x(lhs)
        return _composition_impl(f.shape, f.stride, rhs.shape, rhs.stride)
    if is_tuple(rhs):
        if len(rhs) > lhs.rank():
            raise ValueError("too many modes in tiler")
        return make_layout(*[composition(lhs.mode(i), rhs[i]) for i in range(len(rhs))])
    if is_underscore(rhs):
        return lhs
    f = coalesce_x(lhs)
    return _composition_impl(f.shape, f.stride, rhs, 1)


def _complement_prefix(shape, stride):
    shape, stride = list(flatten(shape)), list(flatten(stride))
    R = len(shape)
    res_s, res_d = [], [1]
    for _i in range(R - 1):
        min_d = min(stride)
        k = stride.index(min_d)
        new_s = min_d // res_d[-1]
        if new_s == 0:
            raise ValueError("Non-injective Layout detected in complement.")
        res_s.append(new_s)
        res_d.append(min_d * shape[k])
        del shape[k], stride[k]
    new_s = stride[0] // res_d[-1]
    if new_s == 0:
        raise ValueError("Non-injective Layout detected in complement.")
    res_s.append(new_s)
    return tuple(res_s), tuple(res_d), stride[0] * shape[0]


def complement(layout: Layout, cotarget=None) -> Layout:
    """cute::complement. Without cotarget the result keeps a terminal size-1 mode(..., _1) whose
    stride defines the complement on the extended domain (pinned-commit semantics)."""
    f = filter_layout(layout)
    fs, fd = f.shape, f.stride
    if not is_tuple(fd) and fd == 0:                         # irreducible rank-1 stride-0
        if cotarget is None:
            return Layout(1, 1)
        return Layout(product(cotarget), 1)
    res_s, res_d, new_d = _complement_prefix(fs, fd)
    if cotarget is None:
        res = coalesce(Layout(res_s, res_d))
        if res.size() == 1:
            return Layout(1, new_d)
        rs = res.shape if is_tuple(res.shape) else (res.shape,)
        rd = res.stride if is_tuple(res.stride) else (res.stride,)
        return Layout(rs + (1,), rd + (new_d,))
    rest_s = product(ceil_div(cotarget, new_d))              # coalesce(static shape) == product
    return coalesce(Layout((res_s, rest_s), (res_d, new_d)))


def slice_tuple(c, t):
    """underscore.hpp slice(c, t): keep elements of t paired with _ (flattening one level per tuple)."""
    if is_tuple(c):
        out = ()
        for ci, ti in zip(c, t):
            out += _lift_slice(ci, ti)
        return out
    return t if is_underscore(c) else ()


def _lift_slice(c, t):
    if is_tuple(c):
        out = ()
        for ci, ti in zip(c, t):
            out += _lift_slice(ci, ti)
        return out
    return (t,) if is_underscore(c) else ()


def slice_layout(c, layout: Layout) -> Layout:
    """Layout(c) with underscores: the sub-layout of the _ modes (offset is crd2idx(c) with _ -> 0)."""
    return Layout(slice_tuple(c, layout.shape), slice_tuple(c, layout.stride))


def zip_tuples(a, b):
    """cute::zip(a, b) on IntTuples: transpose ((a0,a1,...),(b0,b1,...)) -> ((a0,b0),(a1,b1),...);
    if a is not a tuple the result is ((a, b),)."""
    if is_tuple(a):
        return tuple((x, y) for x, y in zip(a, b))
    return ((a, b),)


def zip2_by(t, guide):
    """tuple_algorithms.hpp zip2_by: ((A,a),((B,b),(C,c)),d) by (X,(X,X)) -> ((A,(B,C)),(a,(b,c),d))."""
    if is_tuple(guide):
        split = [zip2_by(t[i], guide[i]) for i in range(len(guide))]
        return (tuple(s[0] for s in split), tuple(s[1] for s in split) + tuple(t[len(guide):]))
    if not (is_tuple(t) and len(t) == 2):
        raise ValueError("zip2_by needs a rank-2 mode at each guide terminal")
    return t
# ---------------------------------------------------------------- end shared core---------------

def logical_divide(layout: Layout, tiler) -> Layout:
    """logical_divide(A, B) -> ((tile), (rest)) per the rules in the module docstring."""
    if isinstance(tiler, Layout):
        # complement(B, |A|) 列出 B 没碰到的那些位置的“步长结构”：先在 tile 内走（B），走完一个 tile
        # 再按 complement 跳到下一个 tile。make_layout(B, B*) 是一个覆盖 [0, |A|) 的双射，
        # 再用 A 复合，就把“逻辑上第 i 个元素”映射回 A 的物理位置。
        # cotarget 用 shape(coalesce(A))，不是 size(A)：两者乘积相同，但 tuple 版本的 ceil_div
        # 在不整除时会逐模式取整（见 core 的 ceil_div），和真实 CuTe 一致。
        cotarget = coalesce(layout).shape
        return composition(layout, make_layout(tiler, complement(tiler, cotarget)))
    if is_tuple(tiler):
        # 按模式切：transform_layout 会把 A 多出来的模式原样追加（composition 的 by-mode版本会丢掉它们）。
        return transform_layout(layout, tiler, lambda l, t: logical_divide(l, t))
    if is_underscore(tiler):
        return layout
    return logical_divide(layout, Layout(tiler, 1))


def _tile_unzip(layout: Layout, tiler) -> Layout:
    # zip2_by 只在 tiler 是 tuple 时重排；tiler 是单个 Layout/int 时 logical_divide 已经是 (tile, rest)。
    guide = _guide(tiler)
    return Layout(zip2_by(layout.shape, guide), zip2_by(layout.stride, guide))


def _guide(tiler):
    if is_tuple(tiler):
        if any(is_underscore(t) for t in _flatten_tiler(tiler)):
            raise ValueError("zipped_divide: tiler contains _, no (tile, rest) pair to zip")
        return tuple(_guide(t) for t in tiler)
    if is_underscore(tiler):
        raise ValueError("zipped_divide: tiler contains _, no (tile, rest) pair tozip")
    return 0


def _flatten_tiler(tiler):
    """Leaves of a tiler tuple (Layouts, ints and _ are leaves)."""
    if is_tuple(tiler):
        out = []
        for t in tiler:
            out += _flatten_tiler(t)
        return out
    return [tiler]


def zipped_divide(layout: Layout, tiler) -> Layout:
    """((tile_0, tile_1, ...), (rest_0, rest_1, ..., extra modes))."""
    return _tile_unzip(logical_divide(layout, tiler), tiler)


def logical_product(block: Layout, tiler) -> Layout:
    """(block, complement(block) o tiler); by-mode for tuple tilers."""
    if isinstance(tiler, Layout):
        # complement(block) 描述“block 之外”的位置：先填 block 的空洞，再越过 block 往后走。
        # 用 tiler 去复合它，就得到“第 k 份 block 的起点”。拼起来：(block 内, 第几份)。
        return make_layout(block, composition(complement(block), tiler))
    if is_tuple(tiler):
        return transform_layout(block, tiler, lambda l, t: logical_product(l, t))
    if is_underscore(tiler):
        return block
    return logical_product(block, Layout(tiler, 1))


def append_rank(layout: Layout, R: int) -> Layout:
    """append<R>(layout): pad with 1:0 modes up to rank R."""
    r = layout.rank()
    if r >= R:
        return layout
    s = layout.shape if is_tuple(layout.shape) else (layout.shape,)
    d = layout.stride if is_tuple(layout.stride) else (layout.stride,)
    return Layout(s + (1,) * (R - r), d + (0,) * (R - r))


def _rank_matched_product(block: Layout, tiler: Layout) -> Layout:
    R = max(block.rank(), tiler.rank())
    # 整体 product（tiler 是一个 Layout，不是 tuple），再按维度配对。补的 1:0 模式让 rank 不同的
    # block 和 tiler 也能逐维配对（见 golden 里 blocked_4_2x3 的 (_1,_3):(_0,_8)）。
    return logical_product(append_rank(block, R), append_rank(tiler, R))


def blocked_product(block: Layout, tiler: Layout) -> Layout:
    """zip(block, tiles): each dimension becomes (block_i, tile_i) — whole blocks side by side."""
    p = _rank_matched_product(block, tiler)
    b, t = p.mode(0), p.mode(1)
    return Layout(zip_tuples(b.shape, t.shape), zip_tuples(b.stride, t.stride))


def raked_product(block: Layout, tiler: Layout) -> Layout:
    """zip(tiles, block): each dimension becomes (tile_i, block_i) — blocks interleaved."""
    p = _rank_matched_product(block, tiler)
    b, t = p.mode(0), p.mode(1)
    return Layout(zip_tuples(t.shape, b.shape), zip_tuples(t.stride, b.stride))


def right_inverse(layout: Layout) -> Layout:
    """CuTe right_inverse: R with layout(R(i)) == i for every i < size(R)."""
    c = coalesce(layout)
    lst = list(c.stride) if is_tuple(c.stride) else [c.stride]
    lsh = list(c.shape) if is_tuple(c.shape) else [c.shape]
    pre = [1]
    for s in lsh:
        pre.append(pre[-1] * s)
    order = sorted(range(len(lst)), key=lambda i: lst[i])
    rs, rd, curr = [1], [0], 1
    for i in order:
        if lst[i] == curr:
            rs.append(lsh[i])
            rd.append(pre[i])
            curr = lsh[i] * lst[i]
    return coalesce(Layout(tuple(rs), tuple(rd)))


def with_shape(layout: Layout, shape) -> Layout:
    """Layout::with_shape(shape) == composition(layout, make_layout(shape))."""
    return composition(layout, Layout(shape))


def swizzle(B: int, M: int, S: int, x: int) -> int:
    """CuTe Swizzle<B,M,S> (S >= 0 here): XOR bits [M+S, M+S+B) down onto bits [M,M+B)."""
    mask = ((1 << B) - 1) << (M + S)
    return x ^ ((x & mask) >> S)


class Swizzled:
    """composition(Swizzle<B,M,S>, base) — the swizzled SMEM layout."""

    def __init__(self, B: int, M: int, S: int, base: Layout):
        self.B, self.M, self.S, self.base = B, M, S, base

    def __call__(self, i):
        return swizzle(self.B, self.M, self.S, self.base(i))

    def size(self):
        return self.base.size()

    def __repr__(self):
        return f"Sw<{self.B},{self.M},{self.S}> o _0 o {self.base!r}"


# ---------------------------------------------------------------- B2: your code ----------------


def max_common_vector(a, b) -> int:
    raise NotImplementedError("max_common_vector")


def max_common_layout(a, b) -> Layout:
    raise NotImplementedError("max_common_layout")


def brute_vector(a, b, size: int) -> int:
    raise NotImplementedError("brute_vector")


def vector_bytes(n_elems: int, elem_bytes: int = 2, cap: int = 16) -> int:
    raise NotImplementedError("vector_bytes")


def alignment_bytes(byte_offsets, cap: int = 16) -> int:
    raise NotImplementedError("alignment_bytes")


_UNIVERSAL = {2: "uint16_t", 4: "uint32_t", 8: "uint64_t", 16: "uint128_t"}


def copy_atom(nbytes: int, async_: bool = False) -> str:
    raise NotImplementedError("copy_atom")


def max_alignment(layout) -> int:
    raise NotImplementedError("max_alignment")


def auto_vector_bits(a, b, elem_bits: int = 16, max_vec_bits: int = 128) -> int:
    raise NotImplementedError("auto_vector_bits")


def copy_plan(a, b, vals_per_thread: int, elem_bits: int = 16, base_offsets_bytes=(0,), async_: bool = False):
    raise NotImplementedError("copy_plan")
