lede: 把 Track A 的五个映射第一次变成真实的 Tensor Core kernel：cp.async 每线程一次 16 B 把 A、B 搬进 swizzle 过的 SMEM，TiledMMA 让 8 个 warp 用 mma.sync 算 128×128 的输出块，累加器留在寄存器里直到 K 走完。只有一级 SMEM（还没有流水线），目的是先把“映射 → 真实 CuTe类型 → 正确结果”这条线跑通。主机部分把三次 partition 和 SMEM 地址用整数复刻，和真实CuTe 对拍。
# F2 · 第一个 Tensor Core GEMM：TiledCopy + swizzle SMEM + TiledMMA

## 题面速览

- 背景知识：[从零设计 Tiled GEMM](../knowledge/from_scratch.html) 的五个映射；[BM、BN、BK 各管什么](../knowledge/tile_dims.html) 的图 2（128 B swizzle 和 ldmatrix 的 bank 组）；[端到端例子](../knowledge/e2e.html) 第 4 节（同样的映射在 64×32 tile 上的坐标表）。上机准备见 [F1](../hints/f1_simt_cute_gemm.html)。
- 设计已经给出：`gpu/tiled_gemm_config.hpp` 里的 `cfg::Config<128,128,64,1,2,4>` ——SMEM layout、G2SCopy、TiledMma、S2RAtom 四个类型。kernel 和 oracle 都 include 它，所以 oracle 打印的坐标就是 kernel 真正用的。
- 你要写：

1. `mine/f2_tensorcore_gemm.cu` 的四个 `DRILL_TODO` ：① `local_tile` ；② G2SCopy 的 `partition_S / partition_D` ；③ TiledMma 的 `partition_A / partition_B` （直接对 SMEM）、`partition_C` 、`make_fragment_C` ；④ K 循环（cp.async → fence → wait → sync → `gemm` → sync）和写回。
2. `mine/f2_tensorcore_gemm.py` ：`copy_starts / copy_coords` 、`smem_offset / bank_group` 、`mma_c_coords` 、`smem_bytes` 、`mma_per_ktile` 。

- 关键规则：

1. 拷贝：线程排成 (32,8):(8,1)，每线程一次 8 个 BF16 = 16 B；(128,64) 要 4 轮，起点 `(t // 8 + 32i, 8(t % 8))` 。
2. SMEM：`smem_offset(r, k) = 512·(r // 8) + 64·j + 8·((r % 8) ⊕ j) + k % 8` ，j = k // 8。
3. MMA：warp w 的 (wm, wn) = (w % 2, w // 2)；lane 的 4 个累加器在行 `lane // 4` （+8）、列 `2(lane %` `4)` （+1）；自然覆盖 32×32，在 128×128 上沿 M、N 各重复 4 次。

测试检查：8 个线程的 copy partition、9 个线程的 MMA partition、整个 (128,64) 的 SMEM 偏移表都和真实 CuTe 一致 · 每 8 行的每个 16 B 块落在 8 个不同的 bank 组 · 覆盖不重不漏 · 计数 · `.cu` 类型检查 · GPU 上 1024³ 和 4096³ 对拍。

## 直觉与类比

::: callout analogy 换上冲压机，但还是“一箱一箱”地干
F1 的小组用普通机床；F2 换成了 Tensor Core 这台冲压机：8 个班组（warp）各管一片 64×32 的零件，每次冲压（mma.sync）一个 warp 一起出 16×8×16 的结果。料从仓库搬上工作台用了自动叉车的前身 cp.async（每个工人一次搬 16 B，不经过自己的工具台），工作台上的料按 swizzle 错开摆，取料时 8 个人不会挤在同一个抽屉前。但现在仍然是“搬完一箱 → 全体等 → 冲压 → 全体等 → 再搬下一箱”，冲压机一半时间在等料——F3 用流水线解决。
:::

## 图解：五个映射在 kernel 里的位置

::: flow 五个映射在 F2 kernel 里的位置
cfg cw=200 rh=86 nw=165
n g 0 0.5 "全局 A, B" "① local_tile" k=dark
n cp 1 0.5 "② G2SCopy" "cp.async 16 B" k=acc
n s 2 0.5 "③ SMEM" "Swizzle<3,3,3>" k=teal
n pa 3 0 "④ partition_A/B" "LDS → 寄存器" k=vio
n acc 3 1 "⑤ 累加器" "64 个 FP32/线程" k=pass
n c 4 0.5 "全局 C" "copy 写回" k=dark
e g -> cp
e cp -> s
e s -> pa
e pa -> acc "mma.sync"
e acc -> c
cap: `gemm(mma, tCsA, tCsB, tCrC)` 收到的是 SMEM 张量时，CuTe 会自己建寄存器 fragment、用普通 LDS 搬过去再调 mma（[[cutlass:include/cute/algorithm/gemm.hpp@gemm|gemm.hpp]] 的 SMEM 版 Dispatch [5]）。F3 会把这一步换成 ldmatrix，并和计算重叠。
:::

## 逐步演示：kernel 的四段 TODO

::: steps 从 local_tile 到写回
::: flow
cfg cw=170 rh=80 nw=145
n lt 0 0 "① local_tile" "gA gB gC" k=dark
n cp 1 0 "② G2SCopy" "partition_S/D" k=acc
n mm 2 0 "③ TiledMma" "partition_A/B/C" k=vio
n kl 3 0 "④ K 循环" "搬 等 算 等" k=teal
n wb 4 0 "写回" "copy" k=pass
e lt -> cp
e cp -> mm
e mm -> kl
e kl -> wb
:::
--- step ① local_tile {hl=lt}
和 F1 完全一样，只是 tiler 变成 (128,128,64)：gA (128,64,K/64)、gB (128,64,K/64)、gC (128,128)。
--- step ② G2SCopy：谁搬哪 16 B {hl=cp}
`auto thr = g2s.get_slice(threadIdx.x); tAgA = thr.partition_S(gA); tAsA = thr.partition_D(sA);`。tAgA 的形状 ((8,1),4,1,k)：第一维是“一条 cp.async 的 8 个值”，第二维是 4 轮。partition_S 和 partition_D 用同一套分工，所以 `copy(g2s, tAgA(_,_,_,kt), tAsA)` 就是每个线程发 4 条 16 B 的 cp.async，源和目的一一对应。
--- step ③ TiledMma：谁持有哪些操作数和累加器 {hl=mm}
`thr_mma.partition_A(sA)` 给出这个线程在 mma 里要用的 A 元素（在 SMEM 里的位置），`partition_C(gC)` 给出它负责的 64 个输出在全局 C 里的位置，`make_fragment_C(tCgC)` 建同形状的 FP32 寄存器数组。形状 ((2,2),4,4)：一个 atom 4 个值 × M 方向 4 次 × N 方向 4 次。
--- step ④ K 循环 {hl=kl}
`copy(g2s, …)` 两次 → `cp_async_fence()`（把刚发的 cp.async 打成一组）→ `cp_async_wait<0>()`（等本线程的所有组落地）→ `__syncthreads()`（等所有线程）→ `gemm(mma, tCsA, tCsB, tCrC)`（4 个 k16 步）→ `__syncthreads()`。
--- step 写回 {hl=wb}
`copy(tCrC, tCgC)`：每个线程写 64 个 float，每对相邻的两个（列 +0/+1）在内存里连续。
:::

## 从小例子开始：先在草稿纸上手算

线程 t = 37：warp 1、lane 5 → (wm, wn) = (1, 0)。

| 问题 | 怎么算 | 结果 |
|---|---|---|
| 4 条 cp.async 的起点 | (37 // 8 + 32i, 8·(37 % 8)) | (4,40) (36,40) (68,40) (100,40) |
| (4, 40) 在 SMEM 的偏移 | j = 5，512·0 + 64·5 + 8·(4 ⊕ 5) + 0 | 328 |
| 第一个累加器 | 行 16·1 + 5 // 4 = 17，列 8·0 + 2·(5 % 4) = 2 | (17, 2) |
| 它的 4 个 atom 值 | (17,2) (17,3) (25,2) (25,3) |  |
| M、N 方向的重复 | 行 +32、列 +32 | (49,2) … (113,98) |

每个 CTA 每个 k-tile 发 (128/16)·(128/8)·(64/16) = 512 条 mma.sync，每个 warp 64 条；每条约 32 个时钟（[执行模型](../knowledge/execution.html)），每个 SMSP 上有 2 个 warp 轮流用同一个 Tensor Core，一个 k-tile 约 2 × 64 × 32 = 4096 clk——而一次显存往返要 1000–1500 clk，F2 每个 k-tile 都要完整地等一次。

## 上机验证

```
./check.sh f2 -s
python3 tools/cu_drill.py run mine/f2_tensorcore_gemm.cu 4096 4096 4096
```

和 F1 比，Tensor Core 让速度上一个台阶（本仓库在 5090 上的实测：1024³ 约 47 TFLOPS，4096³ 约 106 TFLOPS，F1 在 4096³ 约 32）；但离 cuBLAS 还很远：每个 k-tile 要等一次完整的访存延迟，SMEM → 寄存器走的是逐元素的 LDS。用 Nsight Compute 看：`smsp__warp_issue_stalled_barrier` （等 `__syncthreads` ）和`long_scoreboard` （等访存）会很高。

## 提示

::: hint 1 先打印形状
`if (thread0()) { print(tAgA); print("\n"); print(tCsA); print("\n"); print(tCrC); print("\n"); }` 。期望：tAgA `((_8,_1),_4,_1,K/64)` ，tCsA `((_2,_2,_2),_4,_4)` ，tCrC `((_2,_2),_4,_4)` 。主机部分的 `copy_coords(t)` 就是 tAgA 前三维展开后的坐标。
:::

::: hint 2 思路拆解
所有 partition 都是“先 `get_slice(threadIdx.x)` ，再 `partition_*` ”。G2SCopy 对 gA 用 `partition_S` 、对 sA 用 `partition_D` ；TiledMma 对 sA 用 `partition_A` 、对 sB 用 `partition_B` 、对 gC 用 `partition_C` 。SMEM 只有一级，所以 `make_tensor(..., SmemLayoutA{})(_, _, 0)` 取出那一级。主机部分：`mma_c_coords` 的四层循环顺序是 N 重复（最外）→ M 重复 → 行 +0/+8 → 列 +0/+1（最内）。
:::

::: hint 3 关键问题的答案
为什么 cp\_async\_wait&lt;0>() 之后还要 \_\_syncthreads()？

`cp_async_wait` 只等本线程发的 cp.async。mma 要用的 A 元素是别的线程搬的，所以还要 `__syncthreads()` 等所有线程都 wait 完。反过来，只有 `__syncthreads()` 而没有 wait 也不行：cp.async 是异步的，线程到达 barrier 时它发的拷贝可能还在路上。
:::

## 参考解

::: solution solutions/f2_tensorcore_gemm.py
{{code: solutions/f2_tensorcore_gemm.py}}
:::

::: solution solutions/f2_tensorcore_gemm.cu
{{code: solutions/f2_tensorcore_gemm.cu}}
:::

## 面试官追问 & 对照源码

- 把 `Swizzle<3,3,3>` 去掉（atom 换成普通的 (8,64) 行主序），主机部分的 `bank_group` 测试会怎样失败？GPU 上会变慢多少？（`l1tex__data_bank_conflicts_pipe_lsu_mem_shared` 指标）
- `Tile<32,64,16>` 里的 64 换成 32 会怎样？`mma_c_coords` 会变吗？（提示：不会——它影响的是 F3 里 ldmatrix 能不能一次喂两个 n8。）
- 对照 [[cutlass:include/cute/algorithm/gemm.hpp#L462-L495|gemm.hpp 的 SMEM 版 Dispatch [5]] ]：它在每个 k 上做了什么？这一步就是 F3 要手动展开、换成 ldmatrix 并和 mma 交错的部分。
