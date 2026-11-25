lede: 设计 Tiled GEMM 的第二个问题：128 个线程怎样协作把一个 32×16 的 A tile 从 global memory 搬走。本题从 ThrLayout = (32,4):(4,1) 和 ValLayout = (1,4) 出发，自己推出 (thread, value) → (m, k) 的 Copy TV 映射，和真实 make_tiled_copy 逐项对拍，再数两种连续性：一个 warp 碰到几个 sector、一个线程能发多宽的向量访问。
# A2 · 线程怎样搬数据：从 ThrLayout / ValLayout 推出 Copy TV 映射，数 sector 与向量宽度

## 题面速览

- **背景知识**：先读 [从零设计](../knowledge/from_scratch.html) 的“② 线程怎么搬”。
- **在体系里的位置**：真实 kernel 用 `make_tiled_copy(copy_atom, thr_layout, val_layout)`（[[cutlass:include/cute/atom/copy_atom.hpp@make_tiled_copy|copy_atom.hpp]]）。它内部做的正是本题：`raked_product(thr, val)` 得到 (m,k) → (thread, value)，再用 `right_inverse` 反过来得到 TV layout。SM80 教程的设置见 [[cutlass:examples/cute/tutorial/sgemm_sm80.cu@gemm_tn|sgemm_sm80.cu]]（16×8 个线程，每线程 8 个 half，128 位 `cp.async`）。
- **你要实现**：`thread_coord`、`copy_coords`、`tv_index_map`、`byte_addr`、`vector_elems`、`copy_stats`、`rank_variants`。
- **关键规则**：
  1. 线程布局是“线程坐标 → 线程号”；搬运需要它的逆“线程号 → 线程坐标”，所以必须是双射。
  2. raked：线程 t 的第 v 个值是 `(tm·VM + v % VM, tk·VK + v // VM)`，值在内、线程在外，**值按列主序编号**。
  3. TV 映射的第 `i = t + T·v` 项是 tile 的列主序下标 `m + 32·k`：这就是 CuTe `get_layoutS_TV()` 的完整取值表。
  4. 向量宽度由“线程内部的连续性”决定：按 v 的顺序，w 个值地址连续且对齐；sector 数由“线程之间的连续性”决定。

**验收要点**：四种方案的 tile 都是 32×16 · 每个线程的 (m,k) 与真实 `partition_S` 逐项相同 · TV 取值表与真实 `get_layoutS_TV()` 相同 · 线程布局求逆 · 非双射报错 · 每个元素恰好被搬一次 · 文档例子：线程 5 搬 (1,4)..(1,7) · 值按 colex 编号 · 字节地址 · 四种方案的向量宽度 · 原点不对齐时向量变窄、sector 变多 · 两种连续性的统计 · 方案排名。

::: callout info 第一次出现的术语
- **TV layout**（Thread-Value layout）：把 (线程号, 线程内第几个值) 映射到 tile 坐标的 layout。Copy 的 TV layout 回答“谁搬哪个元素”。
- **ThrLayout / ValLayout**：线程在 tile 上怎么排、每个线程负责的小块形状。两者的乘积覆盖整个 tile。
- **sector**：global memory 访问的最小单位，32 字节。一个 warp 的一条访存指令碰到的 sector 越少，需要的内存事务越少。
- **向量化访问**：一个线程用一条指令读写 8 或 16 字节（例如 `LDG.64`、`LDG.128`），前提是这些字节连续且对齐。
:::

## 直觉与类比

::: callout analogy 搬家公司分工
32×16 的 tile 是 512 箱书，128 个搬运工每人搬 4 箱。“线程之间的连续性”像是一辆货车（一个 sector）能不能一次装下好几个人搬的箱子：大家搬同一排相邻的箱子，一辆车就够；大家各搬一排，就要很多辆车。“线程内部的连续性”像是一个人一次能抱几箱：4 箱挨着放，一次抱走（向量访问）；4 箱分散在不同排，只能跑 4 趟（4 条指令）。
:::

## 图解：从两个小 layout 到一张 TV 表

::: flow make_tiled_copy 的推导
cfg cw=230 rh=96 nw=180
n thr 0 0 "ThrLayout" "(32,4):(4,1)" k=acc
n inv 1 0 "线程号 → 坐标" "求逆：(t//4, t%4)" k=vio
n val 0 1 "ValLayout" "(1,4)" k=acc
n rk 1 1 "raked 组合" "值在内、线程在外" k=vio
n tv 2 0.5 "Copy TV 映射" "(t, v) → (m, k)" k=teal
n st 3 0.5 "两种连续性" "sector · 向量宽度" k=pass
e thr -> inv
e val -> rk
e inv -> tv
e rk -> tv
e tv -> st
cap: CuTe 里同一件事是 `layout_tv = right_inverse(raked_product(thr, val))`（[[cutlass:include/cute/atom/copy_atom.hpp@make_tiled_copy|make_tiled_copy]]）。Track B 会用代数重写这一步，这里先用整数推导。
:::

## 逐步演示

::: steps 同样的 32×16 tile、128 个线程，四种分工
::: flow
cfg cw=210 rh=80 nw=180
n base 0 0 "base" "1 条指令 · 32 sector" k=pass
n col 1 0 "colthr" "1 条指令 · 128 sector" k=warn
n v41 2 0 "t8x16_v4x1" "4 条指令 · 32 sector" k=warn
n v22 3 0 "t16x8_v2x2" "4 条指令 · 64 sector" k=todo
:::
--- step 文档的方案 base {hl=base}
`ThrLayout (32,4):(4,1)`：线程 t 的坐标是 `(t // 4, t % 4)`；`ValLayout (1,4)`：4 个值在同一行。线程 5 搬 (1,4) (1,5) (1,6) (1,7)（真实 CuTe 的 `partition_S` 也是这四个）。每个 warp 32 个线程覆盖 8 整行 × 32 字节 = 8 个 sector；每个线程 8 字节连续 → 1 条指令。4 个 warp 共 32 个 sector，效率 1.0。
--- step 线程列主序 colthr {hl=col}
同样的形状，stride 改成 `(1,32)`：相邻 lane 沿 M 往下走。每个线程仍有 8 字节连续（1 条指令），但一个 warp 的 32 个 lane 落在 32 个不同的行上，每行碰一个 sector：每个 warp 32 个 sector，总共 128 个，效率只有 0.25。**线程内部连续，线程之间不连续。**
--- step 每线程一列 t8x16_v4x1 {hl=v41}
`ThrLayout (8,16):(16,1)`、`ValLayout (4,1)`：每个线程搬一列上的 4 个元素。线程内部不连续，只能 4 条 2 字节指令；但每条指令里整个 warp 覆盖 2 整行，总 sector 数仍是 32。**线程之间连续，线程内部不连续。**
--- step 2×2 小块 t16x8_v2x2 {hl=v22}
`ValLayout (2,2)` 按列主序编号，v0、v1 是上下相邻的两个元素，v0、v2 才在同一行。按值的顺序看没有连续的两个 → 4 条指令；每个 warp 4 条指令共碰 16 个 sector，总共 64 个。排名：`base < colthr < t8x16_v4x1 < t16x8_v2x2`（先比指令数，再比 sector）。
:::

再看一个容易忽略的点：同样是 base 方案，如果 tile 原点 `k0 = 2`（不是 4 的倍数），每组 4 个值的起点只有 4 字节对齐，向量宽度掉到 2，指令数变 2，每个 warp 碰 24 个 sector，总数 96。文档说“每线程 4 个连续 half 是 8 bytes 的连续数据，但最终能否发出某种特定宽度的向量 load/store，还要看地址对齐”，说的就是这种情况。

## 易错点

- **忘了求逆。** ThrLayout 是“坐标 → 线程号”，不是“线程号 → 坐标”。`(32,4):(4,1)` 的线程 5 在 (1,1)，不在 (5,0)。测试 `test_thread_coord_inverts_thread_layout`。
- **值按行主序编号。** CuTe 的值按 colex（列主序）编号。`test_values_are_numbered_colex` 用 t16x8_v2x2 抓这个：v1 是 (1,0)，不是 (0,1)。
- **TV 表的下标顺序。** 第 i 项是 `t = i % T, v = i // T`（线程维最快），不是 t-major。测试 `test_tv_index_map_matches_real_get_layoutS_TV`。
- **只看线程内部连续。** colthr 每个线程 8 字节连续，但 sector 数是 base 的 4 倍。测试 `test_stats_two_kinds_of_contiguity`。
- **忘了对齐。** 连续不等于能发向量指令。测试 `test_misaligned_origin_shrinks_vector`。

## 对照文档

> “这里必须同时检查两种连续性：线程之间的连续性：同一个 warp 的不同 lane 访问的地址是否集中、连续，能否减少 Global Memory transaction。线程内部的连续性：同一个线程负责的多个 value 是否连续、对齐，能否合并成向量化访问。这两个条件不是一回事。……于是我们需要一张表，把‘线程 t 的第 v 个 value’映射到 tile 的矩阵坐标：Copy TV Layout: (t,v) → (m,k)。”

本题的四种方案正好把“两个条件不是一回事”拆成了四个象限：两种都好（base）、只有线程内部好（colthr）、只有线程之间好（t8x16_v4x1）、都不好（t16x8_v2x2）。

## 上机验证

```bash
playground/run.sh playground/a2_copy_tv.cpp     # 真实 CuTe 的 make_tiled_copy
./check.sh oracle a2                            # 重新生成 tests/golden/a2_copy_tv.json
```

本机的真实输出：

```text
tiler     : (_32,_16)
TV layout : ((_4,_32),(_4,_1)):((_128,_1),(_32,_0))
thread   0 : (0,0) (0,1) (0,2) (0,3)
thread   1 : (0,4) (0,5) (0,6) (0,7)
thread   4 : (1,0) (1,1) (1,2) (1,3)
thread   5 : (1,4) (1,5) (1,6) (1,7)
thread 127 : (31,12) (31,13) (31,14) (31,15)
```

读 TV layout：线程号 t 拆成 `(t % 4, t // 4)`，对应列主序下标的 stride `(128, 1)`，即 k 组 ×128（4 列 × 32 行）和 m ×1；值 v 的 stride 是 32（下一列）。所以线程 t 的第 v 个值在下标 `128·(t%4) + t//4 + 32·v`，也就是 `m = t//4, k = 4·(t%4) + v`。golden 里用的是 1 个 half 的 atom，打印成 `((_4,_32),(_1,_4)):((_128,_1),(_0,_32))`：形状不同，取值表完全相同。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
thread_coord((32, 4), (4, 1), 5)               # (1, 1)
copy_coords((32, 4), (4, 1), (1, 4))[20:24]    # [(1, 4), (1, 5), (1, 6), (1, 7)]
copy_stats((32, 4), (1, 32), (1, 4))["total_sectors"]   # 128
copy_stats((32, 4), (4, 1), (1, 4), k0=2)["sectors_per_warp"]   # [24, 24, 24, 24]
rank_variants(VARIANTS)    # ['base', 'colthr', 't8x16_v4x1', 't16x8_v2x2']
```

#### 关键问题

1. ThrLayout 的逆为什么必须存在？什么样的 ThrLayout 不可逆？
2. 为什么一个 warp 的 sector 数是 base 方案的 8，而不是 32？
3. 线程内部有 2 个元素连续（像 t16x8_v2x2 的 v0、v2），为什么还是不能发 4 字节指令？
:::

::: hint 2 思路拆解与常见坑
- `thread_coord`：枚举所有 (tm, tk)，算线程号，存进反查表；检查线程号恰好是 0..T−1 各一次。
- `copy_coords`：两层循环，线程在外、值在内，用 raked 公式。
- `tv_index_map`：先拿到 t-major 的坐标表 `cs`，第 i 项取 `cs[(i % T)·V + i // T]`。
- `vector_elems`：从 w = 8 往下试，检查每个线程每组 w 个值的地址相差 2 字节且组首按 2w 对齐。
- `copy_stats`：每 32 个线程一组，每条指令收集 `[addr, addr + vec_bytes)` 覆盖的 sector 集合。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. 搬运时每个线程都要知道自己的坐标，所以必须能从线程号唯一反推坐标。stride 让两个坐标算出同一个线程号（例如 `(4,4):(2,1)`），或者某些线程号没有坐标，都不可逆。
2. 一个 warp 的 32 个线程是 8 行 × 每行 4 个线程，每行 4 × 8 字节 = 32 字节，正好 1 个 sector，所以 8 个；colthr 每个线程独占一行，就是 32 个。
3. 向量访问取的是“按值顺序相邻”的几个值：CuTe 的 copy atom 一次拿 ValLayout 的前 w 个值，v0 和 v1 上下相邻、地址差 128 字节，所以不能合并。想让 v0、v1 同行，ValLayout 要写成 `(2,2):(2,1)`（行主序编号）。
:::
:::

## 参考解

::: solution solutions/a2_copy_tv.py
{{code: solutions/a2_copy_tv.py}}
:::

## 面试官追问 & 对照源码

- `make_tiled_copy` 返回的 TiledCopy 有一个 `Tiler_MN`。它和 ThrLayout、ValLayout 的 shape 是什么关系？（对照 [[cutlass:include/cute/atom/copy_atom.hpp@make_tiled_copy|make_tiled_copy]] 里 `product_each(shape(layout_mn))`。）
- SM80 教程为什么用 `(16,8):(8,1)` 个线程、每线程 8 个 half？这比本题的 (32,4) × (1,4) 好在哪里？（提示：16 字节 `cp.async`，见 [[cutlass:include/cute/arch/copy_sm80.hpp@SM80_CP_ASYNC_CACHEALWAYS|SM80_CP_ASYNC_CACHEALWAYS]]。）
- 如果 A 是 M-major（列主序）存的，ThrLayout 和 ValLayout 该怎样改，才能让两种连续性都成立？
- sector 数和 transaction 数是一回事吗？L1 的 128 字节 cache line 在这里扮演什么角色？
