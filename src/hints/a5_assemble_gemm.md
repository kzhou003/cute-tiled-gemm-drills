lede: 把 A1–A4 的五个映射首尾相连，写出一个只靠索引映射算出来的 64×64 Tiled GEMM：逐 CTA、逐 K 片地把数据从全局内存搬进 swizzle 过的 SMEM，再按 MMA 的 TV 布局读进寄存器、在累加器里累加、最后写回一次。顺便统计访存 sector、SMEM wavefront 和 mma 条数，并把 C[5][7] 经过的每一层都追踪出来，和真实 CuTe 的 partition 结果对拍。
# A5 · 把五个映射拼成完整的 Tiled GEMM：模拟、统计、追踪 C[5][7]

## 题面速览

- **背景知识**：[从零设计](../knowledge/from_scratch.html) 的“五个空间映射”和 [真实代码导读](../knowledge/real_code.html)。
- **在体系里的位置**：这就是 [[cutlass:examples/cute/tutorial/sgemm_sm80.cu@gemm_device|sgemm_sm80.cu 的主循环]] 去掉流水线之后的样子：`local_tile` → `copy(copy_a, tAgA, tAsA)` → `copy(s2r_atom_a, ...)` → `gemm(mma, tCrA, tCrB, tCrC)`。
- **你要实现**：`wavefronts`、`simulate_gemm`、`trace`。A1–A4 的结果（`local_tile_coords`、`copy_coords`、`smem_offset`、`ptx_a/b/c`、`tiled_tv`）已经给出。
- **关键规则**：
  1. 每个 CTA 一个 (128, 8) 的累加器数组，K 循环 4 次都不清零、不写回；K 全部完成后按 C 的 TV 写回一次。
  2. 每个 K 片先把 A、B 的 (32,16) tile 写进 SMEM（偏移 = `smem_offset(r, c, swizzle)`），再**只从 SMEM** 读进寄存器。
  3. gmem sector = 32 字节；全局字节地址 `2·(行·64 + 列)`。SMEM wavefront = max(同一 bank 不同 word 的最大个数, ⌈字节数 / 128⌉)，同一 word 广播免费。
  4. smem → 寄存器用“32 位 LDS”模型：每个 warp 每个 K 片，A 的 4 个寄存器、B 的 4 个寄存器各一条指令。

**验收要点**：给出的 copy / MMA / swizzle 与真实 CuTe 的 `partition_S`、`partition_A/B/C`、`composition(Swizzle)` 一致 · wavefront 模型的边界例子 · 三种 swizzle 下结果都等于 `A @ B.T` · C 的每个元素恰好写一次 · 每个 CTA 的计数 256 / 64 / 32 · `Swizzle<2,2,2>` 和不 swizzle 一样是 256 个 load wavefront，`Swizzle<1,3,3>` 降到 128 · `trace(5, 7, 3)` 和 `trace(5, 7, 50)` 的完整结果 · trace 与真实 partition 一致 · 越界与形状错误报错。

::: callout info 第一次出现的术语
- **sector**：全局内存访问的最小粒度，32 字节。一条 warp 指令触及的 sector 越少越好。
- **bank / wavefront**：SMEM 有 32 个 bank，每个 bank 每周期服务一个 4 字节 word。一条指令里同一 bank 的不同 word 要分多个 wavefront 串行完成，这就是 bank conflict。
- **LDS**（Load from Shared）：SMEM → 寄存器的普通 load；`LDS.32` 每 lane 读 4 字节。
- **epilogue**：主循环结束后把累加器写回全局 C 的阶段。
:::

## 直觉与类比

::: callout analogy 中央厨房的一条流水线
全局内存是冷库，SMEM 是备菜台，寄存器是每个厨师手边的砧板，累加器是锅。每一轮（K 片）：搬运工（copy 线程）按清单（Copy TV）从冷库把 32×16 的一箱菜搬上备菜台，按台面编号（swizzle）摆好；厨师按菜谱（MMA TV）从台面取到砧板，下锅（`mma.sync`）。锅一直在火上，四轮之后才出锅装盘（epilogue）。搬运工一次搬多少、台面怎么摆，决定了冷库门口排不排队（sector）、备菜台前挤不挤（wavefront）。
:::

## 图解：一个 K 片里数据走过的五层

::: flow CTA (bm, bn) 的一个 K 片
cfg cw=180 rh=100 nw=150
n g 0 0 "全局 A, B" "local_tile → (32,16)" k=dark
n cp 1 0 "Copy TV" "t → 行 t//4" k=acc
n sm 2 0 "SMEM" "Sw<2,2,2> ∘ (32,16)" k=teal
n rg 3 0 "寄存器" "MMA TV_A / TV_B" k=vio
n acc 3 1 "累加器" "(128, 8)，跨 K 留住" k=pass
n c 4 1 "全局 C" "只写一次" k=dark
e g -> cp
e cp -> sm
e sm -> rg
e rg -> acc "mma.sync"
e acc -> c
cap: 每一层的索引都来自前面四题的函数；`simulate_gemm` 里不允许出现 `A[32*bm:..., ...]` 这样的切片直接参与计算，否则就不是在检验映射了。
:::

## 逐步演示

::: steps trace(5, 7, 3)：CTA (0,0)，第 0 个 K 片
::: flow
cfg cw=190 rh=80 nw=160
n kt 0 0 "CTA 与 K 片" "(0,0)，k_tile 0" k=dark
n cp 1 0 "copy" "A：线程 20 的 v=3" k=acc
n sm 2 0 "SMEM" "A：偏移 87" k=teal
n hold 3 0 "MMA 持有者" "(21,1), (85,1)" k=vio
n acc 3 1 "累加器" "(23, 1)" k=pass
e kt -> cp
e cp -> sm
e sm -> hold
e hold -> acc
:::
--- step 哪个 CTA、哪个 K 片 {hl=kt}
`5 // 32 = 0`、`7 // 32 = 0`、`3 // 16 = 0`：CTA (0,0) 的第 0 个 K 片。A[5][3] 在 tile 内是 (5, 3)，B[7][3] 是 (7, 3)。
--- step 谁搬的 {hl=kt,cp}
copy 的逆：行 5、列 3 由线程 `4·5 + 3//4 = 20` 的第 `3 % 4 = 3` 个值搬运。真实 CuTe 的 `partition_S` 给线程 5 的是 `(1,4) (1,5) (1,6) (1,7)`，和公式“行 t//4、列 4(t%4)+v”一致。B[7][3] 由线程 28 的 v=3 搬。
--- step 放在 SMEM 哪里 {hl=cp,sm}
`16·5 + 3 = 83`，第 [4,6) 位是 01，异或到第 [2,4) 位：`83 ^ 4 = 87`。B 的 (7,3)：`115 ^ 12 = 127`。线程 5 写到 `16 17 18 19`：swizzle 以 4 个 half 为一组移动，一个线程的 4 个值仍然连续。
--- step 谁把它读进寄存器 {hl=sm,hold}
A 的 (5,3) 被线程 21 和 85 的第 1 个值持有（N 方向的两个 warp 都要这一行），B 的 (7,3) 被 29 和 61 持有（M 方向的两个 warp）。线程 5 的 MMA 读取的 SMEM 偏移是 `22 23 150 151 30 31 158 159`。
--- step 累加并写回 {hl=hold,acc}
C 的 (5,7) 只由线程 23 的第 1 个累加器持有。CTA (0,0) 一共 `mma = 32` 条；`gmem_sectors = 256`（A、B 各 32×64 个 half = 4096 字节 = 128 个 sector，没有浪费）；`smem_store_wavefronts = 64`（每条 copy 指令 256 字节 = 2 个 wavefront，无冲突）；`smem_load_wavefronts = 256`（128 条 LDS，每条 2 个 wavefront：2 路冲突）。
:::

## 易错点

- **每个 K 片清零累加器。** 结果只剩最后一个 K 片的贡献。测试 `test_simulated_gemm_is_correct`。
- **在 SMEM 用逻辑下标而不是 swizzle 后的偏移。** 写和读必须用同一个 `smem_offset`，否则数值会错；如果写读都忘了 swizzle，数值对但 wavefront 计数不同。测试 `test_swizzle_222_does_not_help_loads_but_133_does`、`test_trace_agrees_with_real_partitions`。
- **wavefront 只数 bank 冲突。** 一条指令读 256 字节时，即使没有冲突也至少 2 个 wavefront。测试 `test_wavefronts_count_bytes_not_only_conflicts`：`[(8i, 8) for i in range(32)]` 是 2。
- **以为 `Swizzle<2,2,2>` 消掉了所有冲突。** 32×16 的 half tile 每行只有 32 字节，`Swizzle<2,2,2>` 只在 4 行内部打乱，第 r 行和第 r+4 行仍落在同一组 bank。

## 对照文档

> “这些映射不是孤立的：Global tile 要由 Copy TV Layout 分配给线程；Copy 目标坐标要映射到 SMEM 地址；SMEM 中的数据要转成 MMA 所需的寄存器 fragment；MMA 输出分布最后还要转换成 Global C 的写回布局。”

本题把这句话里的四次“转换”逐个执行。对文档 swizzle 的一个补充：在本题的 32 位 LDS 模型下，`Swizzle<2,2,2>` 和不 swizzle 都是 256 个 load wavefront；`Swizzle<1,3,3>`（用行号的第 2 位改 word 的第 2 位）才降到 128。swizzle 要和“读的人怎么读”一起设计，这正是 B2、C2 要用 layout 代数去检查的事情。

## 上机验证

```bash
playground/run.sh playground/a5_assemble_gemm.cpp   # 真实 CuTe，本机编译运行
./check.sh oracle a5                                # 重新生成 tests/golden/a5_assemble_gemm.json
```

本机的真实输出：

```text
thread  20 copies (5,0) (5,1) (5,2) (5,3)
           holds A (5,0) (5,1) (13,0) (13,1) (5,8) (5,9) (13,8) (13,9)
thread  21 copies (5,4) (5,5) (5,6) (5,7)
           holds A (5,2) (5,3) (13,2) (13,3) (5,10) (5,11) (13,10) (13,11)
thread  85 copies (21,4) (21,5) (21,6) (21,7)
           holds A (5,2) (5,3) (13,2) (13,3) (5,10) (5,11) (13,10) (13,11)
thread  23 accumulates C (5,6) (5,7) (13,6) (13,7) (5,22) (5,23) (13,22) (13,23)
A(5,2) lives at smem offset 86 (Swizzle<2,2,2>)
```

线程 21 和 85 持有完全相同的 A 片段：它们是 N 方向两个 warp 里同一个 lane。搬运和持有是两张不同的表：线程 20 搬了第 5 行，但它在 MMA 里拿的是另一些元素。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
wavefronts([(4 * i, 4) for i in range(32)])       # 1   每 lane 一个 bank
wavefronts([(0, 4)] * 32)                         # 1   广播
wavefronts([(128 * i, 4) for i in range(32)])     # 32  全部落在 bank 0
wavefronts([(8 * i, 8) for i in range(32)])       # 2   256 字节
C, st = simulate_gemm(A, B)
st[(0, 0)]   # {'gmem_sectors': 256, 'smem_store_wavefronts': 64, 'smem_load_wavefronts': 256, 'mma': 32, ...}
simulate_gemm(A, B, (1, 3, 3))[1][(0, 0)]["smem_load_wavefronts"]   # 128
trace(5, 7, 3)["C"]                                                  # {'tile': (5, 7), 'acc': (23, 1)}
```

#### 关键问题

1. CTA (0,0) 的 gmem sector 为什么正好是 256？什么样的 Copy TV 会让它变大？
2. 不 swizzle 时，一条 A 的 LDS.32 里哪两组 lane 冲突？`Swizzle<2,2,2>` 为什么没帮上忙？
3. trace 里 copy 的逆为什么可以写成一个公式，MMA 的持有者却要搜索（或用 right_inverse）？
:::

::: hint 2 思路拆解与常见坑
- `wavefronts`：把每个字节区间展开成 4 字节 word 的集合（去重 = 广播），按 `word % 32` 分桶，取 max(最大桶, ⌈4·|words| / 128⌉)。
- `simulate_gemm` 的骨架：`for bm, bn → acc = zeros((128, 8)) → for kt in range(4)`：
  1. 每个 warp 一条 A 的 copy、一条 B 的 copy：用 `copy_coords(t)` 得 tile 坐标，全局坐标加 `(32·bm, 16·kt)`，写 `sA[smem_offset(r, c, swizzle)]`；同时收集 sector 和 SMEM 字节区间。
  2. 每个 warp 8 条 LDS：用 `tiled_tv` 得 (t, v) 的 tile 坐标，从 SMEM 读进 `rA[t, v]`、`rB[t, v]`。
  3. 每个 warp 2 条 mma：按 `ptx_a/b/c` 把寄存器拼回 16×16、8×16 的小矩阵，乘完加回 `acc`。最后按 `tiled_tv("C")` 写回 C，并记录写入次数。
- `trace`：CTA 和 K 片用整除；copy 线程是 `4·row + col // 4`；持有者遍历 `tiled_tv` 找偏移 `row + 32·col`。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. A 和 B 每个 K 片各 32×16 个 half = 1024 字节 = 32 个 sector，4 个 K 片两种操作数共 256 个，每个 sector 只被碰一次。如果每个线程只搬 1 个元素、线程沿 M 方向排列（列主序 Copy TV），一条 warp 指令会碰 32 行、每行一个 sector、每个 sector 只用 2 字节：每个 K 片每个操作数要 16 条指令 × 32 个 sector = 512 个，是现在的 16 倍（L1/L2 能合并掉一部分重复，但指令和事务数实打实地多了）。
2. lane (g, q) 读 word `8·g + q`（g 是行），`bank = (8g + q) % 32`，所以 g 和 g+4 同 bank：2 路冲突。`Swizzle<2,2,2>` 只用行号的低 2 位去改列号，g 和 g+4 的低 2 位相同，改法也相同，冲突不变；`Swizzle<1,3,3>` 用行号的第 2 位把 g+4 平移 4 个 word，冲突消失。
3. Copy TV 是双射，`raked_product` 写出来的 `layout_mn` 本身就是它的逆（B1）；MMA 的 A、B 有复制，一个元素有两个持有者，没有唯一的逆，只能列举（或者在 TV 上做 right_inverse 得到“其中一个”持有者）。
:::
:::

## 参考解

::: solution solutions/a5_assemble_gemm.py
{{code: solutions/a5_assemble_gemm.py}}
:::

## 面试官追问 & 对照源码

- 真实 kernel 用 `cp.async` 把 gmem → smem 变成异步，并用多个 SMEM stage 让下一片的拷贝和这一片的 mma 重叠。对照 [[cutlass:examples/cute/tutorial/sgemm_sm80.cu@gemm_device|sgemm_sm80.cu]] 里的 `cp_async_wait<K_PIPE_MAX-2>`：stage 数怎样影响 SMEM 用量？
- 用 `ldmatrix` 代替 LDS.32 时，一条指令每 lane 读 16 字节，wavefront 模型怎样变？`Swizzle<2,2,2>` 在那种读法下是不是够用？（A3 已经回答了一半。）
- 64×64 的问题只有 4 个 CTA。真实 GPU 有一百多个 SM，这样的 tile 大小会浪费什么？Track E 的配置搜索会回答这个问题。
- 把 C 的写回也看成一次 copy：每个线程 8 个 float 分散在 4 行 × 4 列，能不能向量化写回？CUTLASS 的 epilogue 为什么先把累加器写进 SMEM 再整体写出？
