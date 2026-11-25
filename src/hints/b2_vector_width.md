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
