lede: 第四、第五个映射问题：Tensor Core 要的数据怎样进入线程寄存器、结果由谁持有并继续累加。本题只从 PTX 手册里的 groupID / threadID_in_group 两个公式出发，推出 m16n8k16 的 TV_A、TV_B、TV_C，再扩展成 2×2 个 warp 的 TiledMMA，最后让累加器在“寄存器”里跨 4 个 K 片累加，并和真实 CuTe 的 MMA_Traits、get_layoutX_TV、partition_A/B/C 逐个对拍。
# A4 · MMA fragment：从 PTX 规则推出 TV_A / TV_B / TV_C，扩展到 2×2 warp，在寄存器里累加

## 题面速览

- **背景知识**：[从零设计](../knowledge/from_scratch.html) 的第四、第五个映射问题；指令本身见 PTX ISA 的 [mma.sync m16n8k16 fragment 布局](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float)。
- **在体系里的位置**：CuTe 把这条指令的“谁持有谁”写成 [[cutlass:include/cute/atom/mma_traits_sm80.hpp@SM80_16x8x16_F32F16F16F32_TN|MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>]] 里的 `ALayout / BLayout / CLayout`；`make_tiled_mma` 按 AtomLayout (2,2,1) 和 `Tile<32,32,16>` 把它扩展到整个 CTA tile（[[cutlass:include/cute/atom/mma_atom.hpp@make_tiled_mma|make_tiled_mma]]）；kernel 里用的是 `partition_A / partition_B / partition_C`（[[cutlass:examples/cute/tutorial/sgemm_sm80.cu@gemm_device|sgemm_sm80.cu]] 的 `gemm(mma, ...)`）。
- **你要实现**：`group_id`、`thread_in_group`、`ptx_a/b/c`、`atom_tv`、`tiled_tv`、`registers_per_thread`、`fragment_shape`、`holders`、`warp_mma`、`cta_tile_gemm`、`valid_order`。IntTuple 打印函数、`MAPPINGS`、`DEPENDS` 已经给出。
- **关键规则**：
  1. `groupID = lane >> 2`，`threadID_in_group = lane % 4`。A、B、C 的行列公式各不相同，不能共用一张表。
  2. atom 内用列主序偏移：A 为 `m + 16k`，B 为 `n + 8k`，C 为 `m + 16n`；TV 的一维下标是 `t + 32·v`（线程最快）。
  3. TiledMMA：`lane = t % 32`，`wm = (t // 32) % 2`，`wn = t // 64`；B 和 C 在 N 方向每个 warp 重复一次：`v = fv + 4·r`，`n += 8·(wn + 2r)`。
  4. 累加器从第一个 K 片到最后一个 K 片一直留在 (128, 8) 的寄存器数组里，只在最后写回一次。

**验收要点**：groupID 公式与越界报错 · lane 5 的 PTX 坐标 · 三个 atom TV 与真实 `MMA_Traits` 逐项相同 · atom 覆盖整个 tile · 三个 tiled TV 与真实 `get_layoutX_TV` 相同 · 线程 0、5、37、127 的 fragment 与真实 `partition_A/B/C` 相同 · 寄存器数 4/4/8 与真实 fragment shape · A、B 每个元素有 2 个持有者，C 只有 1 个，`holders("C",5,7) == [(23,1)]` · 一条 `warp_mma` 等于 numpy 的 `C + A·Bᵀ` · 32×K 的 CTA tile GEMM 正确且 `mma`、`k_slices`、`acc_stores = 1024` 正确 · 形状错误报错 · 五个映射的依赖顺序。

::: callout info 第一次出现的术语
- **MMA**（Matrix Multiply-Accumulate）：Tensor Core 指令，`D = A·B + C`。**m16n8k16**：一条指令算 16×8 的 C，K 方向 16。
- **lane**：warp 里的线程号 0..31。**fragment**：一个 lane 在寄存器里持有的那几个元素。
- **accumulator**：保存部分和的 C fragment，跨 K 循环一直留在寄存器里。
- **TiledMMA**：多个 MMA atom 按 AtomLayout 排列、再按 Tile 重复，覆盖整个 CTA tile 的 TV 分布。
:::

## 直觉与类比

::: callout analogy 32 人合拼一张拼图
一条 `mma.sync` 是 32 个人（lane）同时拼一张拼图。规则写在说明书（PTX 手册）上：每 4 个人一组（groupID），组里第几个（threadID_in_group）决定你拿哪两列。A 是一张 16×16 的图，每人手里 8 块；B 是 8×16，每人 4 块；结果 C 是 16×8，每人保管 4 块。三张图的分法不同，所以有三张对照表（TV_A、TV_B、TV_C）。4 个小组（warp）排成 2×2 去拼 32×32 的大图时，每个组在 N 方向还要拼两次；做完一轮 K 不交卷，拿着手里的 C 块继续拼下一轮，四轮之后才交卷（写回全局内存）。
:::

## 图解：从 lane 到 tile 坐标

::: flow 一个 (t, v) 怎样变成 C tile 的坐标
cfg cw=240 rh=96 nw=190
n t 0 0.5 "线程 t" "0..127" k=dark
n lane 1 0 "lane = t % 32" "g = lane>>2, q = lane%4" k=acc
n w 1 1 "warp (wm, wn)" "(t//32)%2, t//64" k=vio
n atom 2 0 "atom 坐标" "PTX 行列公式" k=teal
n tile 3 0.5 "tile 坐标" "m+16wm, n+8(wn+2r)" k=pass
e t -> lane
e t -> w
e lane -> atom
e atom -> tile
e w -> tile
cap: 真实 CuTe 的 `thr_layout_vmnk` 是 `(_32,_2,_2,_1):(_1,_32,_64,_0)`：lane 最快，然后是 M 方向的 warp，再是 N 方向的 warp。`v = fv + 4r` 里的 r 是 `Tile<32,32,16>` 带来的 N 方向重复。
:::

## 逐步演示

::: steps lane 5、线程 37、元素 C[5][7]
::: flow
cfg cw=185 rh=80 nw=160
n l5 0 0 "lane 5" "g = 1, q = 1" k=acc
n af 1 0 "A fragment" "8 个 half" k=vio
n t37 2 0 "线程 37 的 C" "wm = 1, wn = 0" k=teal
n hold 3 0 "谁持有 C[5][7]" "唯一：(23, 1)" k=pass
n kl 3 1 "K 循环" "4 片，累加在寄存器" k=warn
e l5 -> af
e af -> t37
e t37 -> hold
e hold -> kl
:::
--- step lane 5 {hl=l5}
`group_id(5) = 1`，`thread_in_group(5) = 1`。真实 CuTe 的 `ALayout = ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))`：线程模式 `(4,8):(32,1)` 正好是“q 乘 32（两列 = 32 个列主序偏移）、g 乘 1（行）”。
--- step A fragment {hl=l5,af}
`[ptx_a(5, i) for i in range(8)] = [(1,2), (1,3), (9,2), (9,3), (1,10), (1,11), (9,10), (9,11)]`。CuTe 的 `ALayout(5, 2) = 41`，`41 % 16 = 9`、`41 // 16 = 2`，即 (9, 2)，和第 2 个值一致。每个 lane 8 个 half = 4 个 32 位寄存器。
--- step 扩展到 2×2 warp {hl=af,t37}
线程 37：lane 5、`wm = 1`、`wn = 0`。真实 `partition_C` 给出它的 8 个累加器坐标：`(17,2) (17,3) (25,2) (25,3) (17,18) (17,19) (25,18) (25,19)`。前 4 个是 atom 坐标的行加 16，后 4 个再在 N 方向加 16（r = 1）。
--- step 持有者 {hl=t37,hold}
`holders("C", 5, 7) = [(23, 1)]`：C 的每个元素只有一个持有者。A 不一样：`holders("A", 9, 2) = [(5, 2), (69, 2)]`，N 方向的两个 warp 都要同一行 A；B 同理 `holders("B", 1, 2) = [(5, 0), (37, 0)]`。这就是文档说的“A/B fragment 可能在多个线程间复制”。
--- step 在寄存器里累加 {hl=hold,kl}
`cta_tile_gemm(A, B)`（A、B 都是 32×64）：每个 K 片 4 个 warp 各发 2 条 `mma.sync`，4 个 K 片一共 `mma = 32` 条；累加器 (128, 8) 从头到尾不离开“寄存器”，最后写回 `acc_stores = 1024` 次，恰好是 32×32 个元素各一次。
:::

## 易错点

- **A、B、C 共用一个行列公式。** A 的行是 `g + 8·((i>>1)&1)`，C 的行是 `g + 8·(i>>1)`：值的编号顺序不同。测试 `test_ptx_examples` 逐个比较 lane 5。
- **把 B 当成 K×N 存。** TN 的 B 以 N×K 存放，本题统一用 (n, k)，偏移 `n + 8k`。测试 `test_atom_tv_matches_mma_traits[B]`。
- **以为 C 也有复制。** A、B 有 2 个持有者，C 只有 1 个。测试 `test_replication_counts`、`test_holders_of_a_element_are_replicated_across_n_warps`。
- **TiledMMA 只按 warp 编号平移。** `Tile<32,32,16>` 让 N 方向每个 warp 重复一次，B 和 C 的每线程值数从 4 变成 8。测试 `test_tiled_tv_matches_get_layoutX_TV`、`test_registers_and_fragment_shapes`（C 的 fragment shape 是 `((_2,_2),_1,_2)`）。
- **每个 K 片写回一次 C。** 测试要求 `acc_stores == 1024`，即只在最后写一次。

## 对照文档

> “因此需要分别描述三套映射：TV_A: (t,v) → (m,k)，TV_B: (t,v) → (n,k)，TV_C: (t,v) → (m,n)。A、B、C 的分布不相同……不能随意把 A/B/C 的 TV Layout 当成同一张表。”
>
> “高性能 kernel 不应该每轮都把部分 C 写回 Global Memory……acc_next = MMA(A_tile, B_tile, acc_current)。”

本题把“三套映射”落成了三个可以和真实 CuTe 逐项对拍的函数，“分布不相同”体现在持有者个数（A/B 2 个，C 1 个）；“累加器留在寄存器”体现在 `acc_stores = 1024`。

## 上机验证

```bash
playground/run.sh playground/a4_mma_fragments.cpp   # 真实 CuTe，本机编译运行
./check.sh oracle a4                                # 重新生成 tests/golden/a4_mma_fragments.json
```

本机的真实输出：

```text
atom A TV: ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
atom B TV: ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
atom C TV: ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
tiled A TV: ((_4,_8,_2,_2),((_2,_2,_2),(_1,_1))):((_64,_1,_16,_0),((_32,_8,_256),(_0,_0)))
A(9,2) is value 2 of thread 5
C(5,7) is value 1 of thread 23
A(9,2) is value 2 of thread 69
partition_fragment_A shape: ((_2,_2,_2),_1,_1)
```

读 CLayout 的值模式 `(_2,_2):(_16,_8)`：第一个 2 是“列 +1”（列主序偏移 +16），第二个 2 是“行 +8”，和 PTX 的 `col = 2q + (i&1)`、`row = g + 8(i>>1)` 一一对应。tiled A TV 里 N 方向 warp 的 stride 是 `_0`：这就是 A 被两个 warp 复制的代数表达。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
[ptx_c(5, i) for i in range(4)]      # [(1, 2), (1, 3), (9, 2), (9, 3)]
atom_tv("A")[:8]                     # [0, 32, 64, 96, 1, 33, 65, 97]
tiled_tv("C")[37::128]               # [81, 113, 89, 121, 593, 625, 601, 633]
holders("A", 9, 2)                   # [(5, 2), (69, 2)]
registers_per_thread()               # {'A': 4, 'B': 4, 'C': 8}
fragment_shape("B")                  # ((2, 2), 2, 1)
cta_tile_gemm(A, B)[1]               # {'mma': 32, 'k_slices': 4, 'acc_stores': 1024}
```

#### 关键问题

1. 为什么 ALayout 的线程模式是 `(4,8):(32,1)` 而不是 `(32):(1)`？
2. 为什么 A 的元素有两个持有者、C 只有一个？如果 AtomLayout 改成 (4,1,1)，A 和 B 的复制个数会怎样变？
3. `fragment_shape("C") = ((2,2),1,2)` 的三个模式分别是什么？
:::

::: hint 2 思路拆解与常见坑
- 先写 `ptx_a/b/c`：检查 `0 ≤ lane < 32`、`0 ≤ i < V`，直接套公式。
- `atom_tv(which)`：`out[t + 32·v] = 列主序偏移(ptx_x(t, v))`。
- `tiled_tv(which)`：对每个 `t < 128`、`v < 8`，先用 `lane, wm, wn` 和 `fv = v % 4, r = v // 4`（A 不拆 v）求 atom 坐标，再平移到 tile；偏移是 `m + 32·k`（A、B）或 `m + 32·n`（C）。
- `holders`：遍历 `tiled_tv`，收集偏移等于目标的 (t, v)。
- `warp_mma`：把 32 个 lane 的片段按 `ptx_x` 放回 16×16、8×16、16×8 的矩阵，算 `C + A·Bᵀ`，再按 `ptx_c` 拆回 (32, 4)。
- `cta_tile_gemm`：累加器 `acc = zeros((128, 8))`；每个 K 片、每个 warp、每个 r，从 A、B 取出该 warp 的片段，调一次 `warp_mma`，结果写回 acc 的对应 4 列；最后按 `tiled_tv("C")` 写回 C。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. 列主序偏移 `m + 16k` 里，lane 的 q 管“两列”（偏移 +32），g 管“行”（偏移 +1）。线程号 `lane = q + 4g` 按 colex 拆成 (q, g)，所以是 `(4,8):(32,1)`：CuTe 不需要分段公式，一个两层的 layout 就把 PTX 规则写完了。
2. AtomLayout (2,2,1) 有两个 N 方向的 warp，它们算的是 C 的不同列，但需要同一行 A，所以 A 被复制 2 份；同理 B 被 M 方向的两个 warp 复制。C 的每个元素只由一个 warp 计算。改成 (4,1,1) 后 A 不再复制、B 被 4 个 warp 复制（当然 tile 形状也要跟着变）。
3. (2,2) 是一条 atom 的 4 个值（列 +1、行 +8），第二个模式 1 是 M 方向的重复次数，第三个模式 2 是 N 方向的重复（`Tile<32,32,16>` 带来的 r）。
:::
:::

## 参考解

::: solution solutions/a4_mma_fragments.py
{{code: solutions/a4_mma_fragments.py}}
:::

## 面试官追问 & 对照源码

- SM80 的路径里 SMEM → 寄存器通常用 `ldmatrix`（`SM75_U32x4_LDSM_N`）。它的 TV layout 和 MMA 的 A fragment 是什么关系？对照 [[cutlass:examples/cute/tutorial/sgemm_sm80.cu@gemm_device|sgemm_sm80.cu]] 的 s2r copy 和 [[cutlass:include/cute/atom/copy_atom.hpp@make_tiled_copy_A|make_tiled_copy_A]]。
- `get_layoutC_TV` 是怎样由 `thrfrg_C` 和线程编号重排构造的？读 [[cutlass:include/cute/atom/mma_atom.hpp@get_layoutC_TV|mma_atom.hpp]]。
- 如果累加器换成 f16（`SM80_16x8x16_F16F16F16F16_TN`），每线程 C 的寄存器数怎样变？精度上要付出什么？
- 每个线程 8 个 f32 累加器只是 32×32 tile 的情况；tile 放大到 128×128、4 个 warp 时，每线程多少个累加器？寄存器压力在什么时候导致 spill？
