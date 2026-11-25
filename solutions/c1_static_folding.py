"""C1 · 编译期化简：常量折叠、coalesce、强度削减与循环不变量外提 — reference solution.

CuTe 的 layout 代数本身不在 GPU 上运行。一个线程算地址 offset = L(tid, v, k) 时，如果 shape 和
stride 都是静态整数（Int<N>），C++ 编译器在实例化模板时就能把整套 layout 代数折叠成几条移位、与运算和加法；如果它们是运行时值，乘法和除法就会留在机器码里。这道题用一个小小的表达式树把
这个过程亲手做一遍：先把“线程的 copy 地址”写成表达式，再依次做常量折叠、强度削减、
coalesce 和循环不变量外提，最后数一数还剩多少运行时运算。

运行例子（与仓库所有题目相同）
    A[64,64] 行主序（K 连续），CTA (0,0) 的 tile (BM,BK) = (32,16)，K 方向 4 轮。
    TiledCopy：ThrLayout (32,4):(4,1)，ValLayout (1,4)；CuTe 算出的 S TV layout 是
        ((_4,_32),(_4,_1)):((_128,_1),(_32,_0))      (tid, v) -> tile 内 colex 索引
    tile 内索引 -> (m, kk)：Layout (32,16) 的 colex 坐标；gmem offset = m*64 + k*16 +kk。
    真实 CuTe 算出的 offset 表在 tests/golden/c1_static_folding.json。

表达式（全部是 Python 元组，结构化比较）
    ('c', n)              常量 n（int）
    ('v', name)           运行时变量（tid、v、k，或动态 shape/stride 的名字，如 'bm'）
    (op, a, b)            op ∈ {'+', '*', '//', '%', '<<', '>>', '&'}；<<, >>, & 的右操作数是常量
    所有变量都假定是**非负整数**（tid、v、k、shape、stride 都满足），所以 // 与 % 是向下取整，
    并且“除以 2^n”可以直接换成右移（真实 C++ 用的是有符号 int，编译器会额外插入符号修正，见讲解页）。

API
    C(n), V(name)                         given：构造常量 / 变量
    evaluate(e, env) -> int               given：env 是 {name: int}
    pretty(e) -> str                      given：中缀打印，便于调试
    layout_expr(stride, coord) -> e       Σ coord_i * stride_i；stride 是扁平 tuple，元素是 int（静态）
                                          或 str（动态变量名）；coord 是同长度的表达式 tuple。
                                          结果从左到右累加：((c0*d0 + c1*d1) + c2*d2)…
    idx2crd_expr(idx, shape) -> tuple     colex 分解（CuTe 约定）：c_0 = idx % s_0，
                                          c_i = (idx // (s_0…s_{i-1})) % s_i，最后一个模式不取模：
                                          c_{n-1} = idx // (s_0…s_{n-2})。shape 元素可以是 int 或 str；
                                          前缀积按从左到右的 '*' 构造（静态部分直接相乘成常量）。
                                          rank 1 时返回 (idx,)。
    copy_offset_expr(static=True) -> e    运行例子的 gmem offset，结构必须是：
                                            tc = idx2crd_expr(V('tid'), (4, 32))
                                            vc = idx2crd_expr(V('v'),   (4, 1))
                                            idx = layout_expr(TV_STRIDE, tc + vc)
                                            m, kk = idx2crd_expr(idx, (BM, BK))
                                            offset = layout_expr((LD, 1), (m, k*BK+ kk))
                                          static=True：TV_STRIDE = (128, 1, 32, 0)，BM, BK, LD = 32, 16, 64
                                          static=False：TV_STRIDE = (4*bm, 1, bm,0)（4*bm 写成 ('*', C(4), V('bm'))），
                                          BM, BK, LD = 'bm', 'bk', 'ld'，k*BK 写成('*', V('k'), V('bk'))。
    fold(e) -> e                          自底向上的常量折叠与恒等化简，直到不再变化：
                                            两边都是常量 → 计算出结果；
                                            x*0 → 0，x*1 → x，x+0 → x，x//1 → x，x%1→ 0，0//x → 0，0%x → 0；
                                            (x*c1)*c2 → x*(c1*c2)；(x//c1)//c2 →x//(c1*c2)（c 都是正常量）；
                                            (x+c1)+c2 → x+(c1+c2)；常量放到 * 和 + 的右边（c*x → x*c，c+x → x+c）。
    strength_reduce(e) -> e               右操作数是 2 的幂常量 2^n（n ≥ 1）时：x*2^n →x<<n，
                                          x//2^n → x>>n，x%2^n → x&(2^n−1)。其他情况不动。
    simplify(e) -> e                      fold 与 strength_reduce 交替做到不动点。
    count_ops(e) -> dict                  对“结构不同的运行时子表达式”计数（相同结构只算一次，模拟 CSE），
                                          常量子树不计；返回 {op: 次数}，只包含出现过的 op。
    coalesce_flat(shape, stride) -> (shape, stride)
                                          静态扁平 layout 的 coalesce（A3 的规则）：去掉尺寸 1 的模式，
                                          s_i*d_i == d_{i+1} 时合并相邻模式；结果仍是 tuple（至少一个模式，
                                          全是尺寸 1 时返回 ((1,), (0,))）。
    layout_index_expr(idx, shape, stride, coalesce=False) -> e
                                          layout_expr(stride, idx2crd_expr(idx, shape))；coalesce=True 时
                                          先 coalesce_flat 再构造。
    hoist(e, loop_var) -> (base, step)    e 必须是 loop_var 的一次函数：e == base + loop_var*step，
                                          base 和 step 都不含 loop_var（先 simplify再分解；
                                          loop_var 只能出现在 '+' 的项里，作为 x、x*c或 x<<n）。
                                          不满足时抛 ValueError。返回的 base、step 都已simplify。
"""
from __future__ import annotations


def C(n: int):
    """常量节点。"""
    return ("c", int(n))


def V(name: str):
    """运行时变量节点。"""
    return ("v", name)


def evaluate(e, env: dict) -> int:
    """在 env 下求值（非负整数语义）。"""
    tag = e[0]
    if tag == "c":
        return e[1]
    if tag == "v":
        return env[e[1]]
    a, b = evaluate(e[1], env), evaluate(e[2], env)
    if tag == "+":
        return a + b
    if tag == "*":
        return a * b
    if tag == "//":
        return a // b
    if tag == "%":
        return a % b
    if tag == "<<":
        return a << b
    if tag == ">>":
        return a >> b
    if tag == "&":
        return a & b
    raise ValueError(f"unknown op {tag!r}")


def pretty(e) -> str:
    """中缀打印。"""
    tag = e[0]
    if tag == "c":
        return str(e[1])
    if tag == "v":
        return e[1]
    return f"({pretty(e[1])} {tag} {pretty(e[2])})"


def _s(x):
    # shape / stride 元素：int → 常量，str → 变量
    return C(x) if isinstance(x, int) else V(x)


def layout_expr(stride, coord):
    """Σ coord_i * stride_i，从左到右累加。"""
    if len(stride) != len(coord):
        raise ValueError("stride and coord must have the same length")
    acc = None
    for c, d in zip(coord, stride):
        term = ("*", c, _s(d))
        acc = term if acc is None else ("+", acc, term)
    return acc


def idx2crd_expr(idx, shape):
    """colex 分解；最后一个模式不取模（CuTe 的约定）。"""
    n = len(shape)
    if n == 1:
        return (idx,)
    out = []
    prefix = None                      # 前缀积表达式（None 表示 1）
    for i, s in enumerate(shape):
        q = idx if prefix is None else ("//", idx, prefix)
        out.append(q if i == n - 1 else ("%", q, _s(s)))
        if prefix is None:
            prefix = _s(s)
        elif prefix[0] == "c" and isinstance(s, int):
            prefix = C(prefix[1] * s)  # 静态部分直接相乘
        else:
            prefix = ("*", prefix, _s(s))
    return tuple(out)


TV_STRIDE_STATIC = (128, 1, 32, 0)


def copy_offset_expr(static: bool = True):
    """运行例子：线程 tid 的第 v 个值在第 k 轮的 gmem offset。"""
    if static:
        tv_stride, BM, BK, LD = TV_STRIDE_STATIC, 32, 16, 64
        kBK = ("*", V("k"), C(16))
    else:
        tv_stride = (("*", C(4), V("bm")), 1, "bm", 0)
        BM, BK, LD = "bm", "bk", "ld"
        kBK = ("*", V("k"), V("bk"))
    tc = idx2crd_expr(V("tid"), (4, 32))
    vc = idx2crd_expr(V("v"), (4, 1))
    # layout_expr 只接受 int/str stride；动态的 4*bm 已经是表达式，单独处理
    coords = tc + vc
    acc = None
    for c, d in zip(coords, tv_stride):
        term = ("*", c, d if isinstance(d, tuple) else _s(d))
        acc = term if acc is None else ("+", acc, term)
    idx = acc
    m, kk = idx2crd_expr(idx, (BM, BK))
    return layout_expr((LD, 1), (m, ("+", kBK, kk)))


def _is_c(e):
    return e[0] == "c"


def _fold_node(e):
    tag = e[0]
    if tag in ("c", "v"):
        return e
    a, b = _fold_node(e[1]), _fold_node(e[2])
    if _is_c(a) and _is_c(b):
        return C(evaluate((tag, a, b), {}))
    # 常量放右边（只对交换的 + 和 *）
    if tag in ("+", "*") and _is_c(a) and not _is_c(b):
        a, b = b, a
    if tag == "*":
        if b == C(0):
            return C(0)
        if b == C(1):
            return a
        if _is_c(b) and a[0] == "*" and _is_c(a[2]):        # (x*c1)*c2
            return ("*", a[1], C(a[2][1] * b[1]))
    if tag == "+":
        if b == C(0):
            return a
        if _is_c(b) and a[0] == "+" and _is_c(a[2]):        # (x+c1)+c2
            return ("+", a[1], C(a[2][1] + b[1]))
    if tag == "//":
        if b == C(1):
            return a
        if a == C(0):
            return C(0)
        if _is_c(b) and b[1] > 0 and a[0] == "//" and _is_c(a[2]) and a[2][1] > 0:# (x//c1)//c2
            return ("//", a[1], C(a[2][1] * b[1]))
    if tag == "%":
        if b == C(1) or a == C(0):
            return C(0)
    return (tag, a, b)


def fold(e):
    """常量折叠 + 恒等化简，直到不动点。"""
    # 为什么要迭代：一次自底向上只能看到“子节点已经化简后的样子”，
    # 像 ((x*2)*4)*1 这样的链需要多轮才能完全合并。
    while True:
        n = _fold_node(e)
        if n == e:
            return e
        e = n


def _pow2(n: int):
    return n >= 2 and (n & (n - 1)) == 0


def strength_reduce(e):
    """乘、除、模 2 的幂 → 移位与按位与。"""
    tag = e[0]
    if tag in ("c", "v"):
        return e
    a, b = strength_reduce(e[1]), strength_reduce(e[2])
    if _is_c(b) and _pow2(b[1]):
        n = b[1].bit_length() - 1
        if tag == "*":
            return ("<<", a, C(n))
        if tag == "//":
            return (">>", a, C(n))       # 只因为变量都非负才合法；有符号 int 需要修正
        if tag == "%":
            return ("&", a, C(b[1] - 1))
    return (tag, a, b)


def simplify(e):
    """fold 与 strength_reduce 交替到不动点。"""
    while True:
        n = strength_reduce(fold(e))
        if n == e:
            return e
        e = n


def count_ops(e) -> dict:
    """结构不同的运行时子表达式按 op 计数（模拟 CSE）。"""
    seen = set()
    counts: dict = {}

    def has_var(x):
        return x[0] == "v" or (x[0] not in ("c", "v") and (has_var(x[1]) or has_var(x[2])))

    def walk(x):
        if x[0] in ("c", "v") or x in seen or not has_var(x):
            return
        seen.add(x)
        counts[x[0]] = counts.get(x[0], 0) + 1
        walk(x[1])
        walk(x[2])

    walk(e)
    return counts


def coalesce_flat(shape, stride):
    """静态扁平 layout 的 coalesce。"""
    modes = [(s, d) for s, d in zip(shape, stride) if s != 1]
    if not modes:
        return (1,), (0,)
    out = [modes[0]]
    for s, d in modes[1:]:
        ps, pd = out[-1]
        if ps * pd == d:                 # 前一个模式走完正好接上
            out[-1] = (ps * s, pd)
        else:
            out.append((s, d))
    return tuple(s for s, _ in out), tuple(d for _, d in out)


def layout_index_expr(idx, shape, stride, coalesce: bool = False):
    """layout(idx) 的表达式；coalesce=True 先合并模式。"""
    if coalesce:
        shape, stride = coalesce_flat(shape, stride)
    return layout_expr(stride, idx2crd_expr(idx, shape))


def _terms(e):
    # 把 '+' 树展平成项列表
    if e[0] == "+":
        return _terms(e[1]) + _terms(e[2])
    return [e]


def _contains(e, name):
    if e[0] == "v":
        return e[1] == name
    if e[0] == "c":
        return False
    return _contains(e[1], name) or _contains(e[2], name)


def hoist(e, loop_var: str):
    """e == base + loop_var*step，base、step 不含 loop_var。"""
    e = simplify(e)
    base, step = None, None
    for t in _terms(e):
        if not _contains(t, loop_var):
            base = t if base is None else ("+", base, t)
            continue
        # 允许的形式：x、x*c、x<<n（x 就是 loop_var，c 不含 loop_var）
        if t == V(loop_var):
            s = C(1)
        elif t[0] == "*" and t[1] == V(loop_var) and not _contains(t[2], loop_var):
            s = t[2]
        elif t[0] == "<<" and t[1] == V(loop_var) and _is_c(t[2]):
            s = C(1 << t[2][1])
        else:
            raise ValueError(f"term {pretty(t)} is not linear in {loop_var}")
        step = s if step is None else ("+", step, s)
    return simplify(base if base is not None else C(0)), simplify(step if step is not None else C(0))
