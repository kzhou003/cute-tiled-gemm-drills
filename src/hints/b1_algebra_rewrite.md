lede: Track A 用手写公式回答了“CTA 拿哪块、线程搬哪些、SMEM 放哪里”。CuTe 不写这些公式，而是用 Layout Algebra 把它们构造出来：local_tile 是 zipped_divide 加切片，Copy TV 是 raked_product 的 right_inverse，SMEM 是 Swizzle 和普通 layout 的 composition。本题用和真实 CuTe 逐字符一致的 Python 代数写出这三个构造，并证明它们和 Track A 的手写公式是同一个函数。
# B1 · 用 Layout Algebra 重写 Track A：zipped_divide、raked_product + right_inverse、Swizzle ∘ Layout

## 题面速览

- **背景知识**：[Layout Algebra 与编译](../knowledge/algebra_compile.html) 的代数操作表。
- **在体系里的位置**：
  - `local_tile` 的定义就是 `zipped_divide` 再切片（[[cutlass:include/cute/tensor_impl.hpp@local_tile|local_tile]]、[[cutlass:include/cute/layout.hpp@zipped_divide|zipped_divide]]）。
  - `make_tiled_copy` 的前三行：`layout_mn = raked_product(thr_layout, val_layout)`，`layout_tv = right_inverse(layout_mn).with_shape(...)`（[[cutlass:include/cute/atom/copy_atom.hpp@make_tiled_copy|make_tiled_copy]]、[[cutlass:include/cute/layout.hpp@raked_product|raked_product]]、[[cutlass:include/cute/layout.hpp@right_inverse|right_inverse]]）。
  - swizzle 过的 SMEM layout 是 `composition(Swizzle<2,2,2>{}, layout)`（[[cutlass:include/cute/swizzle.hpp@Swizzle|Swizzle]]）。
- **你要实现**：`local_tile_view`、`tile_map`、`hand_local_tile`、`make_layout_tv`、`tv_coord`、`owner`、`hand_copy_coord`、`smem_layout_map`、`hand_smem_offset`、`same_function`。整套 Python CuTe 代数（`Layout`、`composition`、`complement`、`coalesce`、`zipped_divide`、`raked_product`、`right_inverse`、`with_shape`、`swizzle` …）已经给出。
- **关键规则**：
  1. `zipped_divide(gA, (32,16)) = ((tile), (rest))`：第 0 个模式是 tile 内，第 1 个模式是“第几个 tile”。切片 `(_, coord)` 得到 tile 的 layout 和一个常数偏移 `rest(coord)`。
  2. `raked_product(thr, val)` 的定义域是 tile 坐标 (m, k)，值是 TV 下标 `t + 128·v`；它的 `right_inverse` 才是 (t, v) → tile 坐标。
  3. `with_shape(L, s) = composition(L, Layout(s))`：只改定义域的形状，不改函数。
  4. tile 坐标的一维偏移按 colex：`m + 32·k`。

**验收要点**：given 的代数核心在 8 个表达式上和真实 CuTe 的打印逐字符相同（这几条从一开始就通过）· `(1,2)` 的 tile layout 与偏移和真实 `local_tile` 相同 · 8 个 (bm, k) 的偏移与真实 rest 模式相同 · 代数版与手写公式相同 · 8 个视图恰好覆盖 A 一次 · 四种 thr/val 的 `layout_mn`、`layout_tv`、tiler 与真实 CuTe 逐字符相同，取值表相同 · `layout_mn ∘ layout_tv` 是恒等 · `tv_coord` 与手写 `(t//4, 4(t%4)+v)` 是同一个函数 · `owner` 是 copy 的逆 · SMEM 映射与真实 `Sw<2,2,2> o _0 o (_32,_16):(_16,_1)` 相同 · 手写位运算与之相同 · `same_function` 能发现差异。

::: callout info 第一次出现的术语
- **zipped_divide**：`logical_divide` 之后把各维的“tile 内”模式拼成第 0 个模式、“tile 编号”模式拼成第 1 个模式。
- **raked_product**：把 val 的每一维插到 thr 对应维的内侧，线程和值交错排列（rake = 耙子：齿和齿之间有空隙）。
- **right_inverse(L)**：满足 `L(R(i)) = i` 的 layout，CuTe 通过对 stride 排序构造它。
- **ComposedLayout**：`Sw<2,2,2> o _0 o L` 读作“先算 L，加偏移 0，再做 swizzle”。
:::

## 直觉与类比

::: callout analogy 同一份座位表的三种写法
Track A 像是手写了一份音乐会的座位表：“第 5 排第 3 座给谁”。代数的写法是给出生成规则：“把大厅按 32×16 的区块切开”（divide）、“每 4 个连号座位给一个人，人沿着排先坐满”（raked_product 再取逆）、“每排的座位号按排号的两位打乱”（composition 一个 swizzle）。生成规则可以和别的规则拼接、化简、编译期检查；而同一个大厅、同一批人，两种写法得到的座位表必须一模一样，这一题就是在逐个座位核对。
:::

## 图解：三个构造

::: flow Track A 的手写公式 ↔ CuTe 的代数构造
cfg cw=360 rh=90 nw=230
n h1 0 0 "CTA tile 偏移" "(32bm+r)·64 + 16k + c" k=warn
n a1 1 0 "zipped_divide + 切片" "((32,16),(2,4))" k=acc
n h2 0 1 "Copy TV" "(t//4, 4(t%4)+v)" k=warn
n a2 1 1 "right_inverse(raked)" "((4,32),4):((128,1),32)" k=vio
n h3 0 2 "SMEM 偏移" "16r+c，异或 [4,6)→[2,4)" k=warn
n a3 1 2 "Swizzle ∘ Layout" "Sw<2,2,2> o (32,16)" k=teal
e h1 -> a1 "same_function"
e h2 -> a2 "same_function"
e h3 -> a3 "same_function"
cap: 右边三个都是真实 CuTe 打印出来的 layout；左边是 A1–A3 手写的公式。测试在每个定义域点上检查两边相等。
:::

## 逐步演示

::: steps make_tiled_copy 的三行：thr = (32,4):(4,1)，val = (1,4)
::: flow
cfg cw=200 rh=80 nw=170
n rp 0 0 "raked_product" "(m,k) → t+128v" k=acc
n ri 1 0 "right_inverse" "(_4,_128):(_128,_1)" k=vio
n ws 2 0 "with_shape" "(128, 4)" k=teal
n q 3 0 "tv_coord / owner" "两个方向都能查" k=pass
e rp -> ri
e ri -> ws
e ws -> q
:::
--- step raked_product {hl=rp}
`layout_mn = ((_1,_32),(_4,_4)):((_0,_4),(_128,_1))`，和真实 CuTe 相同。读法：m 方向 `(1,32)`，val 的 1 在内侧、thr 的 32 在外侧（stride 4，线程号 +4）；k 方向 `(4,4)`，val 的 4 在内侧（stride 128 = 下一个值）、thr 的 4 在外侧（stride 1 = 下一个线程）。tiler 是每个模式的 size：(32, 16)。`layout_mn(31, 15) = 511`，即 (t, v) = (127, 3)。
--- step 取逆 {hl=rp,ri}
`right_inverse(layout_mn) = (_4,_128):(_128,_1)`：把 stride 排序（1, 4, 128）后反过来写。它满足 `layout_mn(R(i)) = i`，测试 `test_layout_tv_is_right_inverse_of_layout_mn` 对全部 512 个 i 检查。
--- step 改形状 {hl=ri,ws}
`with_shape(R, (128, 4)) = ((_4,_32),_4):((_128,_1),_32)`。线程模式 `(4,32):(128,1)`：线程号的低 2 位乘 128（列 4(t%4)，乘 tile 高度 32），高 5 位乘 1（行 t//4）。真实 `get_layoutS_TV` 是 `((_4,_32),(_4,_1)):((_128,_1),(_32,_0))`，多一个 size 1 的模式，作为函数完全相同。
--- step 两个方向都能查 {hl=ws,q}
`layout_tv(5, 2) = 193`，`193 % 32 = 1`、`193 // 32 = 6`：线程 5 的第 2 个值是 tile 的 (1, 6)，和手写 `(5 // 4, 4·(5 % 4) + 2)` 一致。反过来 `owner(layout_mn, 31, 15) = (127, 3)`。
:::

## 易错点

- **把 zipped_divide 的 rest 偏移当成 tile 编号。** 切片之后剩下的是 `rest(coord)` 这个元素偏移：`(1, 2) → 2048 + 32 = 2080`，不是 `1·4 + 2`。测试 `test_local_tile_view_is_zipped_divide_then_slice`。
- **把 raked_product 当成 TV layout。** 它的定义域是 tile 坐标，方向是反的；直接拿来当 (t, v) → 坐标，线程 5 会得到错位的结果。测试 `test_make_layout_tv_matches_cute`。
- **with_shape 写成新 `Layout(shape, stride)`。** `with_shape` 是 composition，stride 由代数算出来，不能手填。
- **tile 偏移按行主序解码。** `layout_tv` 的值是 colex 偏移 `m + 32·k`。测试 `test_algebraic_copy_tv_equals_hand_formula`。
- **swizzle 异或的方向写反。** 是把第 [4,6) 位异或到第 [2,4) 位：`(1, 0) → 20`。测试 `test_algebraic_smem_equals_hand_formula`。

## 对照文档

> “因为一个高性能 kernel 中，数据要经过很多次‘重新观察’……若每一步都手写索引公式，代码会脆弱、重复，而且很难证明没有漏元素或冲突。Layout Algebra 的作用是把‘映射与映射之间的关系’变成可组合的运算。”

本题把这几句话变成了可以运行的等式：同一个映射，手写一遍、代数构造一遍，在整个定义域上逐点相等；`test_local_tile_views_tile_A_exactly_once` 就是文档说的“证明没有漏元素”。

## 上机验证

```bash
playground/run.sh playground/b1_algebra_rewrite.cpp   # 真实 CuTe，本机编译运行
./check.sh oracle b1                                  # 重新生成 tests/golden/b1_algebra_rewrite.json
```

本机的真实输出：

```text
zipped_divide(gA, (32,16)) = ((_32,_16),(_2,_4)):((_64,_1),(_2048,_16))
rest mode at (1,2) = 2080   (the tile origin of CTA row 1, k-tile 2)
raked_product(thr, val)   = ((_1,_32),(_4,_4)):((_0,_4),(_128,_1))
right_inverse(...)         = (_4,_128):(_128,_1)
layout_tv                  = ((_4,_32),_4):((_128,_1),_32)
tv(5, 0..3) = 129 161 193 225  (column-major index m + 32k in the 32x16 tile)
smem                       = Sw<2,2,2> o _0 o (_32,_16):(_16,_1)
```

given 的 Python 代数核心（`Layout`、`composition`、`zipped_divide`…）也在 golden 里校验过：`test_given_core_matches_cute_printouts` 对 8 个表达式比较真实 CuTe 的打印字符串，从一开始就会通过。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
gA = Layout((64, 64), (64, 1))
zipped_divide(gA, (32, 16))                   # ((_32,_16),(_2,_4)):((_64,_1),(_2048,_16))
local_tile_view(gA, (32, 16), (1, 2))         # ((_32,_16):(_64,_1), 2080)
mn, tv, tiler = make_layout_tv(Layout((32, 4), (4, 1)), Layout((1, 4)))
tiler                                         # (32, 16)
tv_coord(tv, tiler, 5, 2)                     # (1, 6)
owner(mn, 31, 15)                             # (127, 3)
smem_layout_map((2, 2, 2), Layout((32, 16), (16, 1)))[1]    # 20（下标 1 = 坐标 (1, 0)）
```

#### 关键问题

1. `zipped_divide` 结果第 1 个模式的 stride `(2048, 16)` 分别是什么意思？
2. 为什么 `make_tiled_copy` 先算 `raked_product` 再取逆，而不是直接写 TV？
3. `layout_tv` 和真实 `get_layoutS_TV` 打印不同，为什么测试仍然认为它们相同？
:::

::: hint 2 思路拆解与常见坑
- `local_tile_view`：`z = zipped_divide(gA, tiler)`；返回 `(z.mode(0), z.mode(1)(coord))`。
- `tile_map`：`[offset + L(i) for i in range(L.size())]`。
- `make_layout_tv`：三行，正好对应 `make_tiled_copy` 的源码；tiler 是 `layout_mn.shape` 每个模式的乘积。
- `tv_coord`：`off = layout_tv((t, v))`，返回 `(off % tiler[0], off // tiler[0])`。
- `owner`：`i = layout_mn((m, k))`，`T` 是线程数（可以从 `layout_mn.shape` 里 thr 的部分读出来），返回 `(i % T, i // T)`。
- `smem_layout_map`：对 `base(i)` 逐个调 `swizzle(B, M, S, ·)`。
- `same_function`：`all(f(x) == g(x) for x in domain)`。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. rest 模式 (2, 4) 是 (tile 行号 bm, K tile 号 k)。stride 2048 = 32 行 × 64：下一个 CTA 行从第 32 行开始；stride 16：下一个 K tile 在同一行往右 16 个元素。
2. “每线程拿 4 个连续值、线程按某种顺序排开”是在 tile 坐标空间里最容易描述的（这就是 product：把 val 这个小块按 thr 复制）。但硬件需要的是反方向：“第 t 个线程第 v 个值在哪”。`right_inverse` 把容易写的方向自动翻成需要的方向，并且在编译期就验证它是双射。
3. 真实版本多了一个 size 1 的模式 `(_4,_1):(_32,_0)`，它不改变任何点的值。测试比较的是函数值（512 个点），而不是打印字符串。
:::
:::

## 参考解

::: solution solutions/b1_algebra_rewrite.py
{{code: solutions/b1_algebra_rewrite.py}}
:::

## 面试官追问 & 对照源码

- `partition_S` 内部用 `tile2thrfrg(zipped_divide(stensor, Tiler_MN{}), ...)`（[[cutlass:include/cute/atom/copy_atom.hpp@tile2thrfrg|copy_atom.hpp]]）：它是怎样把本题的 `layout_tv` 作用到一个具体 tensor 上的？
- 如果 thr 改成 `(32,4):(1,32)`（线程沿 M 排），`layout_tv` 会变成什么？每线程的 4 个值还连续吗？`test_make_layout_tv_matches_cute[colthr]` 里有答案。
- `right_inverse` 遇到非单射的 layout（例如 stride 0 的广播）会怎样？为什么 MMA 的 A fragment 不能用它求唯一的持有者？
- `local_tile` 的 `Step<_1, X, _1>` 投影在代数里对应什么操作？对照 [[cutlass:include/cute/tensor_impl.hpp#L1050-L1068|带投影的 local_tile]]。
