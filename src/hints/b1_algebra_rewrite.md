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

## 从小例子开始：先在草稿纸上手算

这一题的三个构造（divide、product + inverse、swizzle）都是“对 layout 做运算”。直接看 64×64、128 个线程会被数字淹没。先拿一个小到每个格子都能写出来的 layout 把每个运算手算一遍，再换成测试里的尺寸（数字变大，步骤一模一样），最后写成一般公式。下面的草稿纸就是你自己推这道题时应该写下的东西。

① 小：8 个元素的 layout

一维 `8:1` 切成 4 个一块；二维 `(4,4):(4,1)` 切成 2×2 的块；4 个线程、每人 2 个值；一行 16 个元素做 swizzle。每个数都能手写。

```
logical_divide(8:1, 4) = (4,2):(1,4)
zipped_divide((4,4):(4,1), (2,2))
  = ((2,2),(2,2)):((4,1),(8,2))
thr (2,2):(2,1) · val (1,2) → tile (2,4)
```

② 测试的尺寸

同样的四步，只是数字变大。

```
zipped_divide((64,64):(64,1), (32,16))
  = ((32,16),(2,4)):((64,1),(2048,16))
layout_tv = ((4,32),4):((128,1),32)
Sw<2,2,2>：(1,0) → 20
```

③ 换一组 thr / val

测试 `test_make_layout_tv_generalizes` 换两组配置，代数不变，只有 stride 跟着变。

```
(8,4):(4,1) · (2,2):(2,1) → tile (16,8)
layout_tv = ((4,8),(2,2)):((32,2),
(16,1))
```

④ 一般从 ①–③ 里看出来的规律。

```
tile 的偏移 = Σ 块号 · (块大小 · 原 stride)
TV = 把 (m,k) → t + T·v 这个双射倒过来
swizzle：c' = c XOR 4·(r % 4)（一次换 4
个）
```

**✎** 草稿纸 1：layout 是一个函数，divide 只是把下标重新分组

```text
0) 先会求值。L = (4,2):(1,4)：坐标 (r, c) → r·1 + c·4
      (2,1) → 2·1 + 1·4 = 6
   也可以喂一个整数 i：先按 colex 拆成坐标 (i % 4, i // 4)，再求值
      i = 6 → (2, 1) → 6          L.map() = [0,1,2,3,4,5,6,7]
   换 stride：(4,2):(2,1) 的 (2,1) → 2·2 + 1·1 = 5   map = [0,2,4,6,1,3,5,7]
1) 一维 divide：logical_divide(8:1, 4)
      tiler 4:1 走过 {0,1,2,3}
      complement(4:1, 8) = 2:4         ← “tile 没碰到的部分”怎么走：下一块从 4 开始
      make_layout(4:1, 2:4) = (4,2):(1,4)
      8:1 ∘ (4,2):(1,4) = (4,2):(1,4)  ← 第 0 维 = 块内，第 1 维 = 第几块
   tiler 换成 2:2（隔一个取一个）：
      complement(2:2, 8) = (2,2):(1,4)
      logical_divide(8:1, 2:2) = (2,(2,2)):(2,(1,4))   map = [0,2 | 1,3 | 4,6 | 5,7] → 第 0 块 {0,2}、第 1 块 {1,3}、第 2 块 {4,6}、第 3 块 {5,7}
2) 二维：g = (4,4):(4,1)（行主序，和 gA 一样 K 连续），tiler (2,2)，按模式各切各的
      第 0 维 4:4 ÷ 2 → 4:4 ∘ (2:1, complement(2:1, 4) = 2:2) = (2,2):(4,8)
      第 1 维 4:1 ÷ 2 → (2,2):(1,2)
      logical_divide = ((2,2),(2,2)):((4,8),(1,2))      ← 每一维是 (块内, 第几块)
      zip：把两个“块内”放一起，两个“第几块”放一起
      zipped_divide  = ((2,2),(2,2)):((4,1),(8,2))      ← (tile, rest)
3) 切片 (_, (1,1))：tile 模式保留，rest 模式代入坐标
      rest(1,1) = 1·8 + 1·2 = 10        tile = (2,2):(4,1)
      local_tile_view(g, (2,2), (1,1)) = ((_2,_2):(_4,_1), 10)
      tile_map = [10+0, 10+4, 10+1, 10+5] = [10, 14, 11, 15]
      手写 (2·1 + r)·4 + 2·1 + c：(0,0)→10  (1,0)→14  (0,1)→11  (1,1)→15   ✓ 同一个函数
```

**✎** 草稿纸 2：Copy TV = raked\_product 再取逆；SMEM = swizzle 一下偏移

```text
4) thr = (2,2):(2,1)（4 个线程排成 2×2，行主序），val = (1,2)（每人一行里 2 个相邻值）
      complement(thr) = _1:_4                    ← thr 占满 [0,4)，下一份从 4 开始
      logical_product = (thr, complement(thr) ∘ val) = ((2,2),(1,2)):((2,1),(0,4))
      raked = 每一维 (val 在内侧, thr 在外侧)：
      layout_mn = ((1,2),(2,2)):((0,2),(4,1))    tiler = (1·2, 2·2) = (2, 4)
      layout_mn 的值表（值 i = t + 4·v）：
                k=0    k=1    k=2    k=3
         m=0     0      4      1      5         (0,2)：m → (0,0)，k=2 → (0,1)：0·4 +
1·1 = 1
         m=1     2      6      3      7         → i = 1 → t = 1 % 4 = 1，v = 1 // 4 =
0
      读法：线程 1 的两个值在 (0,2)、(0,3)；手写 (t // 2, 2·(t % 2) + v) 也给出这两个
5) right_inverse(layout_mn)：
      coalesce → (4,2):(2,1)        前缀积 pre = (1, 4, 8)
      stride 从小到大：stride 1（shape 2, pre 4）== curr 1 ✓ 追加 2:4，curr = 2
                       stride 2（shape 4, pre 1）== curr 2 ✓ 追加 4:1，curr = 8
      R = (2,4):(4,1)               检查 layout_mn(R(1)) = layout_mn(4) = 1 ✓
      with_shape(R, (4, 2)) = ((2,2),2):((4,1),2) = layout_tv
      layout_tv(3, 1)：t=3 → (1,1) → 1·4 + 1·1 = 5，v=1 → 1·2 = 2，合计 7
        → (7 % 2, 7 // 2) = (1, 3)；手写 (3 // 2, 2·1 + 1) = (1, 3) ✓
6) Swizzle<2,2,2>：把第 [4,6) 位异或到第 [2,4) 位；base = 16r + c
      (1,0)：x = 16 = 0b01_0000，[4,6) = 01 → 右移 2 → 0b0100 = 4，16 XOR 4 = 20
      (5,7)：x = 87 = 0b101_0111，[4,6) = 01 → 4，87 XOR 4 = 83 = 16·5 + 3
      (2,5)：x = 37 = 0b10_0101，[4,6) = 10 → 8，37 XOR 8 = 45 = 16·2 + 13
      规律：第 [4,6) 位就是 r % 4，于是 c' = c XOR 4·(r % 4)：一行里按 4 个一组换位置
```

下面这张图让你随意改小尺寸的 gA 和 tiler（行主序 `(M,K):(K,1)` ，tiler 必须整除）。左边每格写的是 gA 的偏移 `m·K +` `k` ，颜色是它属于哪个 tile；右边是选中 tile 的视图，每格写的是 `tile_map` 的值，角上的小字是视图下标 i。面板打印的就是 `local_tile_view` / `tile_map` / `hand_local_tile` 对这组参数应该返回的东西。默认参数就是草稿纸第 2、3 步。

::: raw
<figure class="flow snap"><div class="flow-head">手算草稿 · 小 gA 上的 zipped_divide + 切片</div><img src="../figs/b1/b1_new-01.png" alt="手算草稿 · 小 gA 上的 zipped_divide + 切片" loading="lazy" style="width:100%;aspect-ratio:1461/1086"></figure>
:::

## 图解详解：三个映射怎样由代数构造出来

::: callout info 关于这一节的图
这些图是交互版讲解页的静态快照，显示的是默认参数下的状态。文中提到的按钮、滑块和悬停提示在这里不能操作；图上的数字和下方的文字、表格、黑底面板说的是同一组值，可以直接对照你的函数输出。
:::

这一节把整道题画成五张可交互的图，顺序和你要写的函数一致：先确认“layout 就是坐标 → 下标的函数”（图 1），再看 `local_tile_view` 怎样由 zipped\_divide 加切片得到（图 2），然后是 Copy TV 的两半：raked\_product 给出正向表（图 3），right\_inverse 把它倒过来（图 4 小例子、图 5 真实尺寸），最后是 SMEM 的 swizzle（图 6）。每张图下方的面板都把当前状态逐步代入公式，数字和参考解、真实 CuTe 的输出一致。

整道题的骨架只有下面几行，剩下的都是在验证“代数构造 == 手写公式”：

```
z = zipped_divide(gA, (32, 16))                     # 图 2：((tile), (rest))
tile, off = z.mode(0), z.mode(1)(coord)             #       切片 (_, coord)
layout_mn = raked_product(thr, val)                 # 图 3：(m, k) → t + 128·v
layout_tv = with_shape(right_inverse(layout_mn), (128, 4))   # 图 4/5：(t, v) → m + 32·k
smem[i]   = swizzle(2, 2, 2, base(i))               # 图 6：Sw<2,2,2> ∘ (32,16):(16,1)
```

### 图 1 · Layout 就是函数：坐标 → 下标

CuTe 的 layout `Shape:Stride` 是一个函数：把坐标的每个分量乘上对应的 stride 再相加。坐标可以是二维的 (r, c)，也可以是一个整数 i（先按 colex，即第一维最快，拆成坐标）。嵌套的 shape（比如 `(2,2)` 作为一维）也照此递归拆开。左边网格每格写的是 `L(r, c)` ；下面一行是同一个函数按一维下标 i 排开的 `L(i)` ，i = r + R·c。几个预设都是本页后面会出现的 layout，先把它们“看成表”。

::: raw
<figure class="flow snap"><div class="flow-head">L(r, c) 与 L(i)</div><img src="../figs/b1/b1_new-02.png" alt="L(r, c) 与 L(i)" loading="lazy" style="width:100%;aspect-ratio:1461/1112"></figure>
:::

参考：CuTe 文档 [[cutlass:media/docs/cpp/cute/01_layout.md|01_layout.md]] （Shape、Stride、colex 下标），求值本身是 [[cutlass:include/cute/stride.hpp#L60|crd2idx]] 。

### 图 2 · `local_tile_view` ：zipped\_divide 把下标重新分组，切片只剩一个偏移

gA 是 64×64 的行主序矩阵 `(64,64):(64,1)` 。`logical_divide(gA, (32,16))` 按维各切各的：m 维变成 `(32,2): (64,2048)` （块内 32 行、共 2 块，下一块跳 32·64 = 2048），k 维变成 `(16,4):(1,16)` 。`zipped_divide` 再把两个“块内”模式拼成第 0 维、两个“第几块”拼成第 1 维。右边画的就是这次重新分组：数字一个没变，只是括号换了位置。最后切片 `(_, (bm, k))` ：tile 模式原样保留，rest 模式代入坐标，变成一个常数偏移。

点 bm / k 换 tile，或者“▶ 播放”把 8 个 tile 走一遍。悬停左边任何一格，看它的偏移怎样拆成“tile 内 + rest”。

::: raw
<figure class="flow snap"><div class="flow-head">zipped_divide((64,64):(64,1), (32,16)) 与切片</div><img src="../figs/b1/b1_new-03.png" alt="zipped_divide((64,64):(64,1), (32,16)) 与切片" loading="lazy" style="width:100%;aspect-ratio:1461/1297"></figure>
:::

源码对照：[[cutlass:include/cute/layout.hpp#L1596-L1625|logical_divide]] （就是 `composition(layout, make_layout(tiler, complement(tiler, shape(coalesce(layout)))))` 这一行）、[[cutlass:include/cute/layout.hpp#L1647-L1656|zipped_divide]] （`tile_unzip(logical_divide(...))` ）、[[cutlass:include/cute/layout.hpp#L1271-L1287|complement]] ；tensor 版的 [[cutlass:include/cute/tensor_impl.hpp#L1030-L1045|local_tile]] 只是在这之上切片。文档：[[cutlass:media/docs/cpp/cute/02_layout_algebra.md#L381-L481|02_layout_algebra.md · Division (Tiling)]] ，里面的 1-D / 2-D 例子和草稿纸第 1、2 步是同一类算式。

### 图 3 · `raked_product` ：正向表 (m, k) → t + 128·v

先回答“tile 里第 (m, k) 个元素由谁搬”。`raked_product(thr, val)` 先算 `logical_product(thr, val) = (thr, complement(thr) ∘ val)` ：thr 占满 [0, 128)，complement 是 `_1:_128` ，所以 val 的第 v 个值被放到 `128·v` 。然后把每一维写成 (val 在内侧, thr 在外侧)。结果 `layout_mn` 的值是 `i = t + 128·v` 。

下图每格写的是线程号 t，颜色是 v。注意一个线程的 4 个值（4 种颜色）在 k 方向相邻，线程沿 k 每 4 列换一个、沿 m 每行跳 4 个：这就是 thr `(32,4):(4,1)` 的行主序排列。切换预设看另外两组配置（测试 `test_make_layout_tv_generalizes` ）。

::: raw
<figure class="flow snap"><div class="flow-head">layout_mn = raked_product(thr, val)</div><img src="../figs/b1/b1_new-04.png" alt="layout_mn = raked_product(thr, val)" loading="lazy" style="width:100%;aspect-ratio:1461/1618"></figure>
:::

源码对照：[[cutlass:include/cute/layout.hpp#L1789-L1803|raked_product]] （`zip(get<1>(result), get<0>(result))` ：rest 在前，所以 thr 落在外侧）、

[[cutlass:include/cute/layout.hpp#L1690-L1698|logical_product]] ；文档 [[cutlass:media/docs/cpp/cute/02_layout_algebra.md#L545-L569|Blocked and Raked Products]] 。

### 图 4 · `right_inverse` ：把箭头倒过来（小例子）

正向表回答“(m, k) 归谁”，但 copy 需要的是反方向：“线程 t 的第 v 个值在哪”。right\_inverse 就是把每条箭头反过来。用 4 个线程 × 2 个值的小例子看清楚：上面一排是 TV 下标 i = t + 4·v，下面是 tile。切到“正向”看 layout\_mn 的箭头（格子 → i），切到“反向”看 layout\_tv 的箭头（i → 格子）。面板一步步跑 right\_inverse 的算法：coalesce、按 stride 排序、逐个拼接。

::: raw
<figure class="flow snap"><div class="flow-head">right_inverse(layout_mn) 与 with_shape</div><img src="../figs/b1/b1_new-05.png" alt="right_inverse(layout_mn) 与 with_shape" loading="lazy" style="width:100%;aspect-ratio:1461/1519"></figure>
:::

源码对照：[[cutlass:include/cute/layout.hpp#L1295-L1345|right_inverse]] （注释写的就是 `layout(result(i)) == i` ；算法和面板里的步骤一致：coalesce → 前缀积 → 按 stride 排序 → fold）。`with_shape` 是 [[cutlass:include/cute/layout.hpp#L206|Layout::with_shape]] ，等于 `composition(*this, make_layout(shape))` 。

### 图 5 · `tv_coord` / `owner` ：真实尺寸的双向查表

换回测试的 128 线程 × 4 值。选一个线程 t 和值 v，图里高亮它的 4 个格子；面板把 `layout_tv((t, v))` 逐项算出来，再解码成 (m, k) = (off % 32, off // 32)，并与手写 `(t // 4, 4·(t % 4) + v)` 和 `owner` 对照。悬停任意格子，则是反方向的 owner 查询。默认状态 (5, 2) 就是测试里的 `tv_coord(tv, tiler, 5, 2) == (1, 6)` 。

::: raw
<figure class="flow snap"><div class="flow-head">layout_tv：(t, v) → (m, k)，owner：(m, k) → (t, v)</div><img src="../figs/b1/b1_new-06.png" alt="layout_tv：(t, v) → (m, k)，owner：(m, k) → (t, v)" loading="lazy" style="width:100%;aspect-ratio:1461/1996"></figure>
:::

源码对照：[[cutlass:include/cute/atom/copy_atom.hpp#L495-L506|make_tiled_copy]] 的三行（raked\_product → right\_inverse.with\_shape → product\_each）就是你的 `make_layout_tv` 。

### 图 6 · `smem_layout_map` ：Swizzle ∘ Layout

SMEM tile 是 32×16 的行主序 `(32,16):(16,1)` ，先算普通偏移 x = 16r + c，再做 `Swizzle<2,2,2>` ：把 x 的第 [4,6) 位异或到第 [2,4) 位。因为 [4,6) 位正好是 r % 4，[2,4) 位正好是“本行第几个 4 元素组”，所以效果就是 第 r 行的 4 个组按 j → j XOR (r % 4) 换位置，组内 4 个元素的顺序不变。每格写的是 swizzle 后的偏移，颜色是它落在本行的第几个组（物理位置）。下方条带画出选中那一行在内存里的样子。切换 B = 0 看没有 swizzle 的对照（测试里的 `smem_A_plain` ）。

::: raw
<figure class="flow snap"><div class="flow-head">Sw&lt;B,2,2> ∘ (32,16):(16,1)</div><img src="../figs/b1/b1_new-07.png" alt="Sw&lt;B,2,2> ∘ (32,16):(16,1)" loading="lazy" style="width:100%;aspect-ratio:1461/1566"></figure>
:::

源码对照：[[cutlass:include/cute/swizzle.hpp#L55|Swizzle]] （`apply` 就是 `offset ^ shiftr(offset & yyy_msk, msk_sft)` ，头部注释画了 YY / ZZ 两组位）、[[cutlass:include/cute/swizzle_layout.hpp#L316-L325|composition(Swizzle, Layout)]] （返回 `ComposedLayout` ，打印成 `Sw<2,2,2> o _0 o (_32,_16):(_16,_1)` ）。

### 速查：代数构造 ↔ 手写公式

| 函数 | 代数构造 | 手写公式（Track A） | 运行例子 |
|---|---|---|---|
| `local_tile_view(gA, tiler, coord)` | `z = zipped_divide(gA, tiler)` ；返回 `(z.mode(0), z.mode(1)(coord))` | 偏移 = bm·(32·64) + k·16 | (1,2) → `((_32,_16): (_64,_1), 2080)` |
| `tile_map(L, off)` | `[off + L(i) for i inrange(size)]` | — | (0,0) → [0, 64,128, …] |
| `hand_local_tile(bm, k)` | — | `(32bm + r)·64 + 16k + c` ，i = r + 32c | 与 tile\_map 逐项相同 |
| `make_layout_tv (thr, val)` | `mn = raked_product(thr, val)` ；`tv = with_shape(right_inverse(mn), (\|thr\|, \|val\|))` ；tiler = 每个模式的 size | — | `((_4,_32),_4): ((_128,_1),_32)`，(32, 16) |
| `tv_coord(tv, tiler, t, v)` | `off = tv((t, v))` ；`(off % 32, off // 32)` | `hand_copy_coord = (t // 4, 4(t % 4) + v)` | (5, 2) → 193 → (1,6) |
| `owner(mn, m, k)` | `i = mn((m, k))` ；`(i % 128, i // 128)` | (4m + k // 4, k % 4) | (31, 15) → 511 → (127, 3) |
| `smem_layout_map(swz, base)` | `[swizzle(B, M, S, base(i))]` ，i = r + 32c | `hand_smem_offset = x ^ ((x >> 2) & 0b1100)` ，x = 16r + c | (1, 0) → 20 |
| `same_function( f, g, D)` | `all(f(x) == g(x) for x in D)` ：在整个定义域上逐点比较，而不是比较打印字符串 |  |  |

三条规律：divide = “块内 + 第几块”，块的偏移是 块号 × (块大小 × 原 stride)；TV = 先在 tile 坐标里写出容易描述的正向表，再用 right\_inverse 倒过来；swizzle = 普通偏移之后再加一层位运算，粒度 2^M = 4 个元素。

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
`right_inverse(layout_mn) = (_4,_128):(_128,_1)`：先 coalesce 成 `(_128,_4):(_4,_1)`，再按 stride 从小到大（1, 4）把对应的模式依次拼起来，新 stride 取该模式在定义域里的步长（前缀积 128、1）。它满足 `layout_mn(R(i)) = i`，测试 `test_layout_tv_is_right_inverse_of_layout_mn` 对全部 512 个 i 检查。
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
