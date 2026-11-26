lede: A2 里“每个线程一次搬 4 个 half”是拍脑袋定的。CuTe 用 max_common_vector 从 gmem 视图和 smem layout 推出“两边同时连续”的最长元素段，再和 max_alignment 推出的对齐取 gcd，得到真正的向量宽度，然后挑 copy atom。本题实现这条推导（含 swizzle 的 2^M 上限），用不依赖代数的暴力定义交叉验证，并看清 padding 为什么让“连续”和“对齐”给出不同的答案。
# B2 · 一次能搬多宽：max_common_vector、swizzle 上限、对齐与 copy atom

## 题面速览

- **背景知识**：[Layout Algebra 与编译](../knowledge/algebra_compile.html) 的 inverse 与 `max_common_vector` 两行。
- **在体系里的位置**：[[cutlass:include/cute/layout.hpp@max_common_vector|max_common_vector]] 和 [[cutlass:include/cute/layout.hpp@max_common_layout|max_common_layout]] 是普通 layout 的版本；一边是 swizzle 时走 [[cutlass:include/cute/swizzle_layout.hpp@max_common_vector|swizzle_layout.hpp 的重载]]，结果和 2^M 取 min。对齐由 [[cutlass:include/cute/layout.hpp@max_alignment|max_alignment]] 给出；`copy(src, dst)` 不带 atom 时走 [[cutlass:include/cute/algorithm/copy.hpp@AutoVectorizingCopyWithAssumedAlignment|AutoVectorizingCopyWithAssumedAlignment]]，向量位宽 = gcd(mcv·元素位宽, 两边的对齐, 128)。copy atom 本身：`UniversalCopy`、[[cutlass:include/cute/arch/copy_sm80.hpp@SM80_CP_ASYNC_CACHEALWAYS|SM80_CP_ASYNC_CACHEALWAYS]]。
- **你要实现**：`max_common_vector`、`max_common_layout`、`brute_vector`、`vector_bytes`、`alignment_bytes`、`copy_atom`、`max_alignment`、`auto_vector_bits`、`copy_plan`。Python CuTe 代数和 `Swizzled(B, M, S, base)` 已经给出。
- **关键规则**：
  1. 普通 × 普通：`common = coalesce(composition(a, right_inverse(b)))`，第 0 个模式 stride 为 1 时它的 size 就是答案，否则 1。
  2. 一边是 `Swizzle<B,M,S>`：先对 base 求，再和 2^M 取 min。两边是同一个 swizzle：不截断。
  3. `max_common_layout` 的值是 tile 内的 colex 下标：这 n 个元素在两边都连续。
  4. “连续”不等于“对齐”：`max_alignment(f) = gcd(size(p 的第 0 模式), p 第 1 模式的 stride)`，`p = logical_divide(f, right_inverse(f))`；真实宽度是两者的 gcd。tensor 起点指针的对齐 layout 看不到，另外取 min。

**验收要点**：10 组 (gmem, smem) 的 `max_common_vector` 与真实 CuTe 相同 · 8 组 `max_common_layout` 与真实 CuTe 逐字符相同 · 参数顺序无关、三种 swizzle 的上限是 2/4/8 · 两个 swizzle 的规则 · 公共 layout 的元素两边都连续 · 暴力定义与代数在 8 组上相同 · padding 一组 mcv 16、`max_alignment` 2、自动向量 32 位 · 9 个 layout 的 `max_alignment` 与真实 CuTe 相同 · 10 组 `auto_vector_bits` 与真实 CuTe 按 copy.hpp 公式算出的位宽相同 · 带对齐要求的暴力定义（封顶 128 位）与自动向量化逐组相等 · 字节数与对齐 · copy atom 名称与非法大小 · 运行例子的 copy plan（含指针偏移 4 字节、K = 60 两种对齐退化）。

::: callout info 第一次出现的术语
- **向量化访存**：一条指令每个线程搬 8 或 16 字节（`LDG.64/128`、`STS.128`、`cp.async` 16B），比逐个 2 字节搬少 4–8 倍指令。前提是两边都连续，并且地址按向量宽度对齐。
- **copy atom**：一种具体的搬运指令。`UniversalCopy<uint64_t>` 是一次 8 字节的普通 load/store；`SM80_CP_ASYNC_CACHEALWAYS<uint128_t>` 是 16 字节的 `cp.async`。
- **padding**：给每行多留几个元素（例如 16 → 18）来错开 bank，是 swizzle 之前常见的老办法。
:::

## 直觉与类比

::: callout analogy 两排储物柜之间搬箱子
gmem 和 smem 是两排编了号的储物柜。一次能推一整辆推车（向量）过去，条件是：这几个东西在两边都放在相邻的柜子里，而且推车停的位置对得上柜门的整数格（对齐）。`max_common_vector` 量的是“两边同时相邻”最多有几格；swizzle 像是对面每 4 格就换一次门牌顺序，所以最多只能保证 4 格；padding 像是对面每排多了两个空柜：一排里仍然有 16 个相邻，但第二排从第 18 格开始，推车只能按 2 格的整数倍停靠——这是 `max_alignment` 量出来的。
:::

## 图解：从两个 layout 到一条指令

::: flow 向量宽度的推导链
cfg cw=200 rh=100 nw=170
n ab 0 0.5 "gmem a, smem b" "(32,16) 的两个 layout" k=dark
n ri 1 0.5 "a ∘ right_inverse(b)" "smem 下标 → gmem 偏移" k=acc
n co 2 0.5 "coalesce 取第 0 模式" "stride 1 才算" k=vio
n sw 3 0 "swizzle 上限" "min(·, 2^M)" k=warn
n al 3 1 "max_alignment" "段起点的 gcd" k=warn
n at 4 0.5 "copy atom" "2/4/8/16 字节" k=pass
e ab -> ri
e ri -> co
e co -> sw
e co -> al
e sw -> at
e al -> at
cap: 两条上限都是 layout 代数算出来的（`max_common_vector` 和 `max_alignment`），在 [[cutlass:include/cute/algorithm/copy.hpp@AutoVectorizingCopyWithAssumedAlignment|copy.hpp]] 里取 gcd。只有 tensor 起点指针本身的对齐 layout 看不到：静态 layout 的 tensor 被假定 128 位对齐，动态的只假定 8 位。
:::

## 从小例子开始：先在草稿纸上手算

先别管 32×16。拿一个 4×4 的 tile，把两个 layout 的 16 个值都写出来，用手找“两边同时连续”的最长段；然后加一个最小的 swizzle 看它怎样把段切断；再加 padding 看“连续”和“对齐”怎样分家。规律清楚以后，再放大到测试的 32×16，最后写成一般公式。

① 小：tile 4×4，gmem a = (4,4):(8,1)

16 个元素，两边的偏移都能手写。smem 换三种：

```
行主序 (4,4):(4,1)  → mcv 4
Sw<1,1,1> ∘ 行主序   → mcv 2
padding (4,4):(6,1) → mcv 4，对齐只有 2
```

② 运行例子：tile 32×16，gmem (32,16):(64,1)

测试里的 A tile（K 连续）对上 5 种 smem。

```
行主序 16 · Sw222 4 · Sw333 8
pad 18：mcv 16、max_alignment 2
M-major：1
```

③ layout 之外的对齐

layout 一样，真实地址却更差：

```
指针偏移 4 B：16 B → 4 B
K = 60：行起点 120·r 字节
  cp.async 16 B → 8 B
```

④ 一般把 ①–③ 里反复出现的东西写成公式。

```
mcv = coalesce(a∘b⁻¹) 第 0 模式（stride
1）
    ∧ 2^M（有 swizzle 时）
位宽 = gcd(mcv·w, 对齐a·w, 对齐b·w, 128)
字节 = min(位宽/8, 每线程值数, 指针对齐)
```

**✎** 草稿纸：tile 4×4，gmem a = (4,4):(8,1)，smem b = (4,4):(4,1)，half（2 字节）

```text
0) 约定    tile 坐标 (m,k)，colex 下标 i = m + 4k（m 最快）。a(m,k) = 8m + k，b(m,k) = 4m + k
1) 把两个 layout 按 i 列出来
   i     :  0  1  2  3 |  4  5  6  7 |  8  9 10 11 | 12 13 14 15
   (m,k) : 00 10 20 30 | 01 11 21 31 | 02 12 22 32 | 03 13 23 33
   a(i)  :  0  8 16 24 |  1  9 17 25 |  2 10 18 26 |  3 11 19 27
   b(i)  :  0  4  8 12 |  1  5  9 13 |  2  6 10 14 |  3  7 11 15
2) right_inverse(b)：smem 偏移 j → 下标 i（满足 b(i) = j）
   j          :  0  1  2  3 |  4  5  6  7 ...
   i = rinv(j):  0  4  8 12 |  1  5  9 13          right_inverse(b) = (_4,_4):(_4,_1) a(rinv(j)) :  0  1  2  3 |  8  9 10 11          ← j = 0..3 时 a = j；j = 4 时 a = 8 ≠ 4
   coalesce(a ∘ rinv(b)) = (_4,_4):(_1,_8)：第 0 模式 _4:_1，stride 1 → mcv = 4
   max_common_layout = rinv ∘ (_4:_1) = _4:_4 → 下标 0, 4, 8, 12 = 第 0 行 (0,0)…(0,3) 3) 不用代数，暴力核对（brute_vector）：smem 里按 n 对齐的块，在 gmem 里也必须是对齐的连续块
   n = 4：块 [4,8) 存 (1,0)…(1,3)，gmem 偏移 8, 9, 10, 11：起点 8 % 4 = 0 ✓，连续 ✓（另外三块同理）
   n = 8：块 [0,8) 存 gmem 0, 1, 2, 3, 8, 9, 10, 11 → 第 4 个就断了 ✗      →
brute_vector = 4 ✓
4) 换成 Sw<1,1,1>：x ↦ x ^ ((x & 0b100) >> 1)，即第 2 位是 1 时翻转第 1 位（低 M = 1 位不
动）
   b(1,0) = 4 = 0b100 → 0b110 = 6     b(1,1) = 5 → 7     b(1,2) = 6 → 4     b(1,3) = 7 → 5
   smem 块 [4,8) 里存的 gmem 偏移：10, 11, 8, 9 → 两两连续，四个不连续
   mcv = min(mcv(a, 行主序), 2^M) = min(4, 2) = 2      brute_vector 也是 2 ✓
5) 换成 padding (4,4):(6,1)：b(m,k) = 6m + k，smem 地址 4、5 是空洞，行起点 0, 6, 12, 18
   right_inverse 只覆盖从 0 开始的连续段 0..3：_4:_4 → mcv 仍然 = 4
   max_alignment(b)：logical_divide(f, rinv f) = (_4,_4):(_1,_6) → gcd(4, 6) = 2
   brute_vector = 2：n = 4 时块 [8,12) 里 10、11 不存在（第 1 行在 6..9）✗
6) 位宽与 atom（每线程 4 个值，指针 16 B 对齐）。gmem 一侧：max_alignment(a) = gcd(4, 8) = 4（行起点 8r）
   auto_vector_bits = gcd(mcv·16, gcd(4·16, ma(b)·16, 128))        bytes = min(位宽/8, vector_bytes(4) = 8, 16)
   行主序        gcd(64, gcd(64, 256, 128) = 64) = 64 位   → 8 B →
UniversalCopy<uint64_t>
   Sw<1,1,1>     gcd(32, gcd(64,  32, 128) = 32) = 32 位   → 4 B →
UniversalCopy<uint32_t>
   pad 6         gcd(64, gcd(64,  32, 128) = 32) = 32 位   → 4 B →
UniversalCopy<uint32_t>
```

下面这张图就是上面的草稿纸，可以换 gmem / smem 和要试的宽度 n。上面是 tile（每格写下标 i 和两边的偏移），中间是 gmem 地址条（每格写这个元素在 smem 的地址），下面是 smem 地址条（每格写这个元素在 gmem 的地址），颜色 = tile 的行 m。smem 条下方的括号是 `brute_vector` 要检查的“按 n 对齐的块”：绿色 ✓ 表示这块在 gmem 里也是对齐的连续块，红色 ✗ 就让这个 n 失败。悬停任意格子，三处会同时高亮同一个元素。

::: raw
<figure class="flow snap"><div class="flow-head">手算草稿 · 4×4 tile 的两条地址条</div><img src="../figs/b2/b2_new-01.png" alt="手算草稿 · 4×4 tile 的两条地址条" loading="lazy" style="width:100%;aspect-ratio:1461/1687"></figure>
:::

## 图解详解：从 layout 推出向量宽度

::: callout info 关于这一节的图
这些图是交互版讲解页的静态快照，显示的是默认参数下的状态。文中提到的按钮、滑块和悬停提示在这里不能操作；图上的数字和下方的文字、表格、黑底面板说的是同一组值，可以直接对照你的函数输出。
:::

这一节把推导链拆成四张图，每张对应一组要实现的函数。默认状态都是测试里的运行例子：A tile 在 gmem 是 `(32,16): (64,1)` （K 连续，ld = 64），smem 有六种选择。

```
mcv   = max_common_vector(gA, sA)        # 图 1：coalesce(a ∘ right_inverse(b)) 的第 0 模式；
align = max_alignment(gA), max_alignment(sA)   # 图 3：段起点能保证的 2 的幂
bits  = auto_vector_bits(gA, sA)         # gcd(mcv·16, align_a·16, align_b·16, 128)
plan  = copy_plan(gA, sA, vals, ...)     # 图 4：再和每线程值数、指针偏移取 min，挑 copy atom
```

### 图 1 · `max_common_vector` 的代数：`a ∘ right_inverse(b)`

按 smem 偏移 j = 0, 1, 2, … 往下数：先用 `right_inverse(b)` 找到 smem 第 j 个位置放的是 tile 的哪个元素 i，再用 a 看它在 gmem 的偏移。只要 a(rinv(j)) == j 一直成立，两边就一起连续；第一次不等的位置就是 mcv。绿色是连续前缀，红色是第一个断点，斜线是 `right_inverse` 的定义域之外（padding 的空洞让它在 16 停下）。swizzle 时这里算的是 base 的 mcv，竖虚线是 2^M 截断。

::: raw
<figure class="flow snap"><div class="flow-head">j → i = rinv(j) → a(i)</div><img src="../figs/b2/b2_new-02.png" alt="j → i = rinv(j) → a(i)" loading="lazy" style="width:100%;aspect-ratio:1461/859"></figure>
:::

源码：[[cutlass:include/cute/layout.hpp#L1440-L1463|max_common_vector （注释里的后置条件 a(b.get_1d_coord(n)) == n 就是这张图的绿色前缀）]]、

[[cutlass:include/cute/layout.hpp#L1420-L1437|max_common_layout]] 、[[cutlass:include/cute/layout.hpp#L1305|right_inverse]] 。

### 图 2 · swizzle 每 2^M 个元素切一刀：`min(·, 2^M)`

swizzle 只改写偏移的第 [M, M+B) 位（用第 [M+S, M+S+B) 位去异或），低 M 位永远不变。所以连续的 2^M 个元素作为一组整体搬家，组内顺序不变；组和组之间的顺序被打乱了。下图是 smem 前 128 个地址（8 排 × 16），每格写“这里放的是 tile 的 (m,k)”，下方横条是两边都连续的段。注意第 0 行：yyy 位全 0，什么都没动，看起来 16 个都连续；但 mcv 必须对每一段都成立，所以只能取 2^M。

::: raw
<figure class="flow snap"><div class="flow-head">Swizzle&lt;B,M,S> ∘ (32,16):(16,1) 的 smem 地址</div><img src="../figs/b2/b2_new-03.png" alt="Swizzle&lt;B,M,S> ∘ (32,16):(16,1) 的 smem 地址" loading="lazy" style="width:100%;aspect-ratio:1461/1097"></figure>
:::

源码：[[cutlass:include/cute/swizzle.hpp#L42-L53|Swizzle 的位图注释（ MBase 个最低位保持不变）]]、[[cutlass:include/cute/swizzle.hpp#L76-L79|apply ： ZZZ ^= YYY]] 、[[cutlass:include/cute/swizzle_layout.hpp#L509-L514|一边是 swizzle ： min(·, 1<<M)]] 、[[cutlass:include/cute/swizzle_layout.hpp#L525-L545|两边都是 swizzle ：相同就不截断]]、[[cutlass:include/cute/swizzle_layout.hpp#L460-L476|max_alignment(Swizzle) = 2^M]] 。

### 图 3 · 连续 ≠ 对齐：`max_alignment`

把 layout 的每一段连续元素（这里是 tile 的每一行）画在一条线性地址轴上。`right_inverse(f)` 覆盖的是从 0 开始的那一段（第 0 行，深色）；`logical_divide(f, right_inverse(f))` 的第 1 个模式的 stride 是下一段的起点。向量宽度 n 要求每一段的起点都是 n 的倍数，所以能保证的对齐 = gcd(段长, 下一段起点)。切换 n，看哪些段起点（▼）通不过。

::: raw
<figure class="flow snap"><div class="flow-head">每一段的起点能保证几个元素的对齐？</div><img src="../figs/b2/b2_new-04.png" alt="每一段的起点能保证几个元素的对齐？" loading="lazy" style="width:100%;aspect-ratio:1461/837"></figure>
:::

源码：[[cutlass:include/cute/layout.hpp#L1940-L1951|max_alignment]] 。K = 60 是同一件事出现在 gmem 一侧：行起点 60·r，`gcd(16, 60) = 4` 个 half = 8 字节。CUDA 对向量访存的对齐要求见 [CUDA C++ Programming Guide · Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)。

### 图 4 · 从位宽到 copy atom：`copy_plan`

最终字节数是几条上限里最小的那条（都是 2 的幂，所以 gcd 就是 min）。前两条来自 layout 代数（`auto_vector_bits` ），后两条是 layout 看不到的：每个线程到底分到几个值、tensor 起点指针本身的对齐。条形图的横轴是字节数（1、2、4、8、16），最短的那根决定结果；下面的表格是能选的 atom。默认状态就是 sgemm\_sm80 的配置：`Swizzle<3,3,3>` + 每线程 8 个 half + `cp.async` 16 B。

::: raw
<figure class="flow snap"><div class="flow-head">copy_plan(a, b, vals_per_thread, base_offsets_bytes, async_)</div><img src="../figs/b2/b2_new-05.png" alt="copy_plan(a, b, vals_per_thread, base_offsets_bytes, async_)" loading="lazy" style="width:100%;aspect-ratio:1461/1247"></figure>
:::

源码：[[cutlass:include/cute/algorithm/copy.hpp#L246-L263|AutoVectorizingCopyWithAssumedAlignment 的两行 gcd]] 、[[cutlass:include/cute/arch/copy.hpp#L46|UniversalCopy]] 、[[cutlass:include/cute/arch/copy_sm80.hpp#L47-L53|cp.async 只收 4/8/16 字节的 static_assert]] 、[[cutlass:examples/cute/tutorial/sgemm_sm80.cu#L357-L372|sgemm_sm80.cu ： Swizzle<3,3,3> 的 smem atom + SM80_CP_ASYNC_CACHEALWAYS<uint128_t>]] 。指令本身：[PTX ISA · cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)（cp-size 只能是 4、8、16；`.cg` 只能 16）。

### 速查

| 函数 | 规则 | 小例子 4×4（a = (4,4):(8,1)） | 运行例子 32×16（a = (32,16):(64,1)） |
|---|---|---|---|
| `max_common_vector` | coalesce(a ∘ rinv(b)) 第 0 模式 stride 1 → 它的 size，否则 1；一边 swizzle：min(·, 2^M)；两边同一个 swizzle：不截断 | 行主序 4 ·Sw&lt;1,1,1> 2· pad 6 → 4· M-major 1 | 行主序 16 ·Sw222 4 ·Sw333 8 · pad 18 → 16 · M-major 1 |
| `max_common_layout` | rinv(b) ∘ 那个第 0 模式；值是 tile 下标；swizzle 截到前 2^M 个 | `_4:_4`（Sw：`_2:_4` ） | `_16:_32`（Sw222：`_4:_32` ） |
| `brute_vector` | 最大的 2 的幂 n：b 里每个对齐的 n 块，在 a 里也是对齐的连续块 | 行主序 4 ·Sw&lt;1,1,1> 2· pad 6 2 ·M-major 1 | pad 18：2（代数 mcv 是 16） |
| `max_alignment` | gcd(size(p 第 0 模式), p 第 1 模式 stride)，p = logical\_divide(f, rinv f)；Swizzled：gcd(2^M,base) | a：4 · pad 6：2 · 行主序：16 | pad 18：2 · 行主序：512 · ld 60：4 ·Sw333：8 |
| `auto_vector_bits` | mcv ≤ 1 → 元素位宽；否则 gcd(mcv·w,gcd(ma(a)·w, ma(b)·w, 128)) | 行主序 64 ·Sw&lt;1,1,1> 32· pad 6 32 ·M-major 16 | 行主序 128 ·Sw222 64 ·pad 18 32 |
| `vector_bytes` | ≤ cap 且整除 n·elem\_bytes 的最大 2 的幂 | `vector_bytes(4) = 8` | `(16) = 16` ·`(3) = 2` |
| `alignment_bytes` | 所有偏移的公共 2 的幂对齐，≤ cap；全 0 → cap | `[0] → 16` | `[0,8,24] → 8`· `[4] → 4` |
| `copy_atom` | 同步 2/4/8/16 B → UniversalCopy&lt;uint16/32/64/128\_t>；cp.async 只有 4/8/16 | 8 B → `UniversalCopy<uint64_t>` | 16 B async → `SM80_CP_ASYNC_CACHEALWAYS< uint128_t>` |
| `copy_plan` | bytes = min(auto/8, vector\_bytes(vals),alignment\_bytes(offs))；vec = bytes / 2 | 行主序, 4 值 → 8 B | Sw333, 8 值,async → 16 B；指针 +4 B |

→ 4 B；K = 60 → 8 B

## 逐步演示

::: steps gmem A tile (32,16):(64,1) 对上 5 种 smem
::: flow
cfg cw=180 rh=80 nw=150
n p 0 0 "smem 行主序" "mcv 16" k=pass
n s2 1 0 "Sw<2,2,2>" "mcv 4" k=warn
n s3 2 0 "Sw<3,3,3>" "mcv 8" k=teal
n pd 3 0 "pad 18" "连续 16，对齐 2" k=todo
n mm 4 0 "smem M-major" "mcv 1" k=dark
:::
--- step 普通行主序 {hl=p}
`right_inverse((32,16):(16,1)) = (16,32):(32,1)`，复合 gmem 后 coalesce 的第 0 个模式是 `16:1`：一行 16 个 half 两边都连续，mcv = 16。真实 CuTe：mcv = 16，mcl = `_16:_32`（tile 下标 0, 32, 64…：同一行的 16 个元素）。
--- step Swizzle<2,2,2> {hl=s2}
swizzle 只改第 M 位以上的位，2^2 = 4 个元素以内保证连续，再长就不保证：mcv = 4，mcl = `_4:_32`。这正是 A2 的配置：每线程 4 个 half = 8 字节，`UniversalCopy<uint64_t>`。
--- step Swizzle<3,3,3> {hl=s3}
上限变成 8：mcv = 8 = 16 字节，可以用 `cp.async` 16B（`SM80_CP_ASYNC_CACHEALWAYS<uint128_t>`，sgemm_sm80.cu 用的就是它）。但每线程得有 8 个连续值：ValLayout 要改成 (1,8)，线程布局随之改成 (32,2) 或 (16,4)。
--- step padding {hl=pd}
`(32,16):(18,1)`：`right_inverse` 只覆盖从 0 开始的连续段 `_16:_32`（一行），所以 mcv 仍然是 16。但下一段从 18 开始：`logical_divide(f, right_inverse(f)) = (_16,_32):(_1,_18)`，`max_alignment = gcd(16, 18) = 2`。自动向量化的位宽 `gcd(16·16, 2·16, 128) = 32`，`copy_plan(g, pad, 8)` 选出 `UniversalCopy<uint32_t>`；带对齐要求的暴力定义也给 2 个 half。
--- step 主序不一致 {hl=mm}
smem 改成 M-major `(32,16):(1,32)`：gmem 里连续的是 K 方向，smem 里连续的是 M 方向，没有一个方向两边都连续：mcv = 1，mcl = `_1:_0`，只能一次 2 字节。这时要么换 smem 主序，要么接受转置的代价（或者用 `ldmatrix.trans` 一类的指令在读的那一侧转置）。
:::

## 易错点

- **swizzle 的上限用 2^B 或 2^S。** 保证不变的是低 M 位，所以是 2^M。测试 `test_swizzled_argument_order_does_not_matter` 用三个 B = M = S 相同的 swizzle 检查，`test_max_common_vector_matches_real_cute[grow_ssw333]` 是 8。
- **两边同一个 swizzle 也截断。** Sw ∘ B0 ∘ rinv(Sw ∘ B1) 里两个 swizzle 抵消，不截断：`test_two_swizzles` 要 512。
- **`max_common_layout` 返回偏移而不是下标。** 它的值是 b 的定义域（tile 内 colex 下标），`_16:_32` 而不是 `_16:_1`。测试 `test_max_common_layout_matches_real_cute`。
- **暴力定义不查对齐。** 不要求 `a(i) % n == 0` 时 `Swizzle<2,2,2>` 会被算成更大的值（第 0 行没被打乱）。测试 `test_brute_force_agrees_with_algebra`。
- **以为 mcv 就是最终宽度。** 还要和 `max_alignment`、每线程值数、指针偏移、atom 支持的大小取 gcd / min。测试 `test_copy_plans_for_the_running_example`：padding 一组 mcv 16、实际 4 字节；K = 60 时 gmem 的 `max_alignment = gcd(16, 60) = 4`，16 字节 `cp.async` 降成 8 字节。
- **整块连续时找不到“下一段”。** 这时 rest 模式是 `_1:_0`，stride 0，gcd(512, 0) = 512，不要当成 0 或报错。测试 `test_max_alignment_matches_real_cute[srow]`。

## 对照文档

> “Copy 不只需要知道‘搬哪个元素’，还要知道能否把相邻元素合并成一次向量化搬运。CuTe 可以利用 source/destination Layout 的映射关系，反向定位并复合，再 coalesce，检查共同连续模式，从而推导两边共同支持的向量宽度。这不等于单凭 Layout 就能保证最终一定发出 128-bit load/store。实际还要考虑地址对齐、指令支持、边界、数据类型和编译器 lowering。”

“反向定位并复合，再 coalesce” 就是 `coalesce(composition(a, right_inverse(b)))`；“还要考虑地址对齐” 在 CuTe 里同样是 layout 代数：`max_alignment`。本题的三个例子：swizzle 把 16 压到 4 或 8；padding 连续 16、对齐只有 2；指针偏移 4 字节或 K = 60 这种 layout 看不全的对齐，要另外取 min。

## 上机验证

```bash
playground/run.sh playground/b2_vector_width.cpp       # 真实 CuTe，本机编译运行
./check.sh oracle b2                                   # 重新生成 tests/golden/b2_vector_width.json
```

本机的真实输出：

```text
plain      mcv =  16   max_alignment(smem) = 512   mcl = _16:_32
Sw<1,1,1>  mcv =   2   max_alignment(smem) =   2   mcl = _2:_32
Sw<2,2,2>  mcv =   4   max_alignment(smem) =   4   mcl = _4:_32
Sw<3,3,3>  mcv =   8   max_alignment(smem) =   8   mcl = _8:_32
pad 18     mcv =  16   max_alignment(smem) =   2   mcl = _16:_32
M-major    mcv =   1   max_alignment(smem) = 512   mcl = _1:_0
```

pad 18 那一行就是“连续不等于对齐”的证据：mcv 是 16，`max_alignment` 只有 2。swizzle 的 `max_alignment` 正好是 2^M，和 mcv 的截断一致。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
g = Layout((32, 16), (64, 1)); s = Layout((32, 16), (16, 1))
max_common_vector(g, s)                         # 16
max_common_layout(g, s)                         # _16:_32
max_common_vector(g, Swizzled(2, 2, 2, s))      # 4
brute_vector(g, Layout((32, 16), (18, 1)), 512) # 2
max_alignment(Layout((32, 16), (18, 1)))        # 2
auto_vector_bits(g, Layout((32, 16), (18, 1)))  # 32
vector_bytes(16), vector_bytes(3)               # (16, 2)
alignment_bytes([0, 8, 24])                     # 8
copy_plan(g, Swizzled(3, 3, 3, s), 8, async_=True)
# {'mcv': 8, 'vec': 8, 'bytes': 16, 'atom': 'SM80_CP_ASYNC_CACHEALWAYS<uint128_t>'}
```

#### 关键问题

1. 为什么是 `composition(a, right_inverse(b))` 而不是 `composition(b, right_inverse(a))`？两者的 mcv 会不同吗？
2. `Swizzle<2,2,2>` 的 mcv 是 4，而 A5 里 `Swizzle<1,3,3>` 能消掉 LDS 冲突：它的 mcv 是多少？这对 copy 意味着什么？
3. padding 为什么让 mcv 和 `max_alignment` 给出不同的答案？
:::

::: hint 2 思路拆解与常见坑
- `max_common_vector`：先分情况（两边都是 Swizzled、只有一边、都不是）。普通情况：`inv = right_inverse(b)`，`common = coalesce(composition(a, inv))`，取第 0 个模式（如果 shape 是元组），stride 为 1 就返回它的 shape。
- `max_common_layout`：同样的 `common` 和第 0 个模式 `m0`，返回 `composition(inv, m0)`；Swizzled 时再 `composition(结果, Layout(2^M))` 截断（结果本来更短就不截）。
- `brute_vector`：建一个“b 值 → 下标”的字典；n 从 2 开始翻倍，对每个 `b(i) % n == 0` 的 i 检查 `a(i) % n == 0` 和后面 n−1 个元素；失败就返回上一个 n。
- `vector_bytes`：从 1 字节开始翻倍，直到超过 cap 或不再整除总字节数。
- `max_alignment`：Swizzled 时 `gcd(2^M, max_alignment(base))`；否则 `f = coalesce(L)`，`p = logical_divide(f, right_inverse(f))`，`gcd(product(p.mode(0).shape), p.mode(1).stride)`。
- `auto_vector_bits`：照抄 copy.hpp 的两行 gcd；mcv ≤ 1 时直接返回元素位宽。
- `copy_plan`：`bytes = min(auto_vector_bits // 8, vector_bytes(vals_per_thread), alignment_bytes(base_offsets_bytes))`。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. `right_inverse(b)` 把 smem 偏移翻回 tile 下标，再用 a 得到 gmem 偏移：结果是“smem 第 j 个连续位置在 gmem 的哪里”，第 0 个模式 stride 为 1 就说明两边一起连续。反过来复合得到的是同一个关系的逆，对双射的情况 mcv 相同；CuTe 选这个方向是因为 b 常常是带 swizzle 的那一边，重载只需处理它。
2. 2^3 = 8。它比 `Swizzle<2,2,2>` 允许更宽的 copy（8 个 half = 16 字节），同时还能消掉 A5 模型里的 LDS.32 冲突；swizzle 的选择要同时看“写的人”（copy 宽度 ≤ 2^M）和“读的人”（bank）。
3. mcv 问的是“从 0 开始有多少个元素两边都连续”，padding 的第一行确实有 16 个；它不问“下一段从哪里开始”。`max_alignment` 问的正是这个：连续段长 16、下一段起点 18，任何一段的起点只能保证是 2 的倍数。CuTe 在 `AutoVectorizingCopyWithAssumedAlignment` 里把两个答案取 gcd；用显式 copy atom 时，选得比这更宽就是你的 bug（运行时地址不对齐会出错）。
:::
:::

## 参考解

::: solution solutions/b2_vector_width.py
{{code: solutions/b2_vector_width.py}}
:::

## 面试官追问 & 对照源码

- 不带 atom 的 `copy(src, dst)` 对静态 layout 假定 128 位对齐，对动态 layout 只假定 8 位（[[cutlass:include/cute/algorithm/copy.hpp@AutoVectorizingCopyWithAssumedAlignment|copy.hpp]]）：为什么动态 stride 的 tensor 默认几乎不向量化？`copy_aligned` 改变了什么？
- `cp.async.cg`（`SM80_CP_ASYNC_CACHEGLOBAL`）只允许 16 字节，`.ca` 允许 4/8/16（[[cutlass:include/cute/arch/copy_sm80.hpp|copy_sm80.hpp]]）。什么时候应该绕过 L1？
- K = 60 时 gmem 行起点 120·r 字节只有 8 字节对齐，本题的 `copy_plan` 把 16 字节 `cp.async` 降成 8 字节。如果 K 是运行时参数（动态 stride），layout 就看不到它了：CUTLASS 的 `AlignmentA` 模板参数在替你承诺什么？
- 把 B 换成 MN-major（列主序）存放时，smem 应该选哪种 layout atom 才能保持 16 字节 copy？对照 SM80 的 `SM80_CP_ASYNC_CACHEALWAYS` + `Swizzle<3,3,3>` 的组合。
