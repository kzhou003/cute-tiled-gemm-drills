lede: 这是 capstone：把 TMA box、SW128 swizzle 的 SMEM 环形缓冲、mbarrier 的 full / empty 两组 barrier、producer / consumer warp group 分工、epilogue 和 PDL 接成一个 Hopper 风格的 GEMM。离散事件模拟器会抓住每一次数据竞争和死锁；结果要和 A @ B.T 逐元素相等，时间线要能看到搬运和计算的重叠。
# E1 · 完整成熟的 Tiled GEMM：TMA → swizzled SMEM 环形缓冲 → mbarrier 流水线 → warp specialization → epilogue → PDL

## 题面速览

- **背景知识**：[时间维度](../knowledge/time_dimension.html) 的 TMA、pipeline、warp specialization 三节，以及 [真实代码导读](../knowledge/real_code.html)。
- **在体系里的位置**：结构对照 [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp|sm90_gemm_tma_warpspecialized_cooperative.hpp]]（1 个 producer warp group + 2 个 consumer warp group）；流水线状态对照 [[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@PipelineState|PipelineState]] 和 [[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@make_producer_start_state|make_producer_start_state]]；主循环对照 [[cutlass:include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp@load|collective 的 load]] 和 [[cutlass:include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp@mma|mma]]；最小的 TMA + mbarrier 主循环见 [[cutlass:examples/cute/tutorial/hopper/wgmma_tma_sm90.cu|wgmma_tma_sm90.cu]]。
- **你要实现**：`PipelineState`（`advance`、`producer_start`、`consumer_start`）、`tma_box_load`、`producer`、`consumer`、`run_gemm`、`chain`。`Barrier`、`Sim`、`sw128`、`smem_read`、`Ctx`、区间工具和所有时间常量已经给出。
- **关键规则**：
  1. `PipelineState.advance`：index 加 1，绕回 0 时翻转 phase；stages == 1 时只翻 phase。producer 从 phase 1 开始，consumer 从 phase 0 开始。
  2. producer 每个 k-tile：等 `empty[s]` → `arrive_expect_tx(full[s], A+B 字节)` → TMA A → TMA B → advance。
  3. consumer 每个 k-tile：等 `full[s]` → MMA（读 stage s）→ `arrive(empty[s])` → advance。`empty` 每个 phase 要 2 次 arrival（两个 consumer WG 都放手才能复用）。
  4. TMA box 总是整块搬：越界元素填 0，字节数按整个 box 算。
  5. PDL 的后继：先只发 B（权重）的 TMA，`grid_wait` 之后再补发 A（前驱的输出）；trigger 在最后一个 k-tile 的 MMA 之后、epilogue 之前。

**验收要点**：given 的 `sw128` 等于真实 CuTe 的 GMMA SW128 atom · `PipelineState` 的序列与真实 CUTLASS 相同、单 stage 只翻 phase、两种起始状态 · TMA box 的 swizzle 位置与 OOB 填零、整块计字节 · producer 的 op 序列（第四个 k-tile 等 phase 0）· PDL producer 在 `grid_wait` 前只发 B、声明 A+B 字节 · consumer 的 op 序列和只有 WG 0 trigger · 五种形状和 stage 数的结果都等于 `A @ B.T`（含边界 tile）· 时间报告（18.3 / 10.6、重叠 4.1、trigger 9.6）· stage 越多越不慢 · 起始时间平移 · PDL 预取权重的收益 · 两个 GEMM 串联有无 PDL 的结果与 spans。

::: callout info 第一次出现的术语
- **TMA**：Hopper 的拷贝引擎。一个线程用描述符发起一次多维 box 拷贝，硬件把 global memory 的一个矩形块搬进 SMEM，越界部分填零。
- **mbarrier**：SMEM 里的异步 barrier。每个 phase 需要固定次数的 arrival，外加 expect_tx 声明的字节全部到达（TMA 落地时 `complete_tx`）。`try_wait.parity(P)` 在“已完成的 phase 数的奇偶 ≠ P”时通过。
- **full / empty barrier**：`full[s]` 表示 stage s 的数据已经到了（producer → consumer），`empty[s]` 表示 stage s 已经用完可以覆盖（consumer → producer）。
- **SW128**：128 字节宽的 swizzle。对 16-bit 元素是 `o ^ ((o >> 3) & 0x38)`，把行号的低 3 位异或进 16 字节块号，让 8 行同一列落在不同 bank。
:::

## 直觉与类比

::: callout analogy 一个厨房的出菜口
出菜口有 stages 个格子。配菜员（producer WG）看到格子空了（empty）就去仓库拿料（TMA），先在格子上挂一张“还差 A、B 两包料”的单子（expect_tx），料到齐单子自动撕掉（full 完成）。两个厨师（consumer WG）看到格子满了就各自做自己那一半（BM/2 行），做完各拍一下格子上的铃（arrive empty），两声铃响格子才算空。格子上贴着一张“第几轮”的奇偶标签（phase）：同一个格子第 n 轮和第 n+1 轮的标签不同，所以没人会把上一轮的状态当成这一轮。配菜员开工时格子都是空的，所以他从“上一轮已经空了”的标签（phase 1）开始等，立刻通过。
:::

## 图解：一个 CTA 里的三个 agent 和两组 barrier

::: flow warp-specialized cooperative GEMM（stages = s）
cfg cw=270 rh=100 nw=180
n p 0 0.5 "producer WG" "empty → expect → TMA" k=acc
n ring 1 0.5 "SMEM ring" "s 个 stage · SW128" k=teal
n c0 2 0 "consumer WG 0" "行 0..63 · MMA · trigger" k=pass
n c1 2 1 "consumer WG 1" "行 64..127 · MMA" k=pass
n epi 3 0.5 "epilogue" "acc → C（裁边）" k=dark
e p -> ring "full[s]" anim
e ring -> c0
e ring -> c1
e c0 -> epi
e c1 -> epi
e c1:b -> p:b "empty[s] × 2" dash
cap: 不变量：producer 写 stage s 之前 `empty[s]` 一定完成过（两个 WG 都读完了），consumer 读 stage s 之前 `full[s]` 一定完成过（两个 box 都落地了）。Sim 的 RAW / WAR 检查就是在验证这两句话。
:::

## 逐步演示

::: steps 128×128×512，stages = 3：8 个 k-tile 怎样在 3 个 stage 里流动
::: flow
cfg cw=180 rh=80 nw=150
n su 0 0 "setup" "0.5 µs" k=dark
n t3 1 0 "TMA k0..k2" "0.5–3.5" k=acc
n m0 2 0 "MMA k0" "2.0–2.6" k=pass
n e0 2 1 "empty[0] 完成" "2.6" k=todo
n k3 3 0 "TMA k3" "等 empty[0]" k=acc
n st 4 0 "稳态" "每 1 µs 一个 tile" k=vio
n end 4 1 "trigger · epi" "9.6 · 10.6" k=warn
e su -> t3
e t3 -> m0
e m0 -> e0
e e0:r -> k3:b
e k3 -> st
e st -> end
:::
--- step setup 与前三个 stage {hl=su,t3}
三个 agent 都先花 0.5 µs 做 setup。producer 的 `PipelineState.producer_start(3)` 是 (index 0, phase 1)：三个 `empty[s]` 都还没完成过 phase，`try_wait(1)` 立刻通过，所以它连续填满 3 个 stage。每个 stage 两个 16 KB 的 box，TMA 引擎每个传 0.5 µs：区间是 0.5–1.0、1.0–1.5 …… 3.0–3.5。
--- step 第一个 MMA {hl=t3,m0}
stage 0 的 B box 在 1.5 传完，再过 0.5 µs 落地：`full[0]` 在 2.0 完成。两个 consumer 的 `consumer_start` 是 phase 0，`try_wait(0)` 在 2.0 通过，并行做 0.6 µs 的 MMA（2.0–2.6），各自只读 A 的一半行。
--- step 释放 stage 0 {hl=m0,e0}
两个 consumer 在 2.6 各 arrive 一次，`empty[0]` 的第一个 phase 完成（需要 2 次）。这正是 empty 每个 phase 需要 `NUM_CONSUMERS` 次 arrival 的原因：只等一个 WG 就复用，另一个 WG 还在读的数据会被 TMA 覆盖（Sim 会报 WAR）。
--- step 第四个 k-tile 复用 stage 0 {hl=e0,k3}
producer 在第 4 个 k-tile 回到 index 0，phase 已经翻成 0：`try_wait(0)` 要求 `empty[0]` 完成过奇数次，2.6 之后成立。TMA k3 在 3.5–4.5 传输（引擎是串行的，而且前面已经排到 3.5）。如果 producer 不翻 phase，它会在 stage 0 上永远等不到，Sim 报 `Deadlock`。
--- step 稳态：搬运是瓶颈 {hl=k3,st}
每个 stage 要 1.0 µs 的 TMA，MMA 只要 0.6 µs：TMA 一直在忙（`tma_busy = 8.0`），MMA 每 1 µs 做一次（2.0、3.0 …… 9.0，`mma_busy = 4.8`）。MMA 和 TMA 的重叠 `overlap = 4.1` µs。
--- step trigger 和 epilogue {hl=st,end}
最后一个 MMA 在 9.6 结束，WG 0 在这里 trigger（`report["trigger"] = 9.6`），然后两个 WG 各做 1.0 µs 的 epilogue，CTA 在 10.6 结束。对比 stages = 1：每个 k-tile 都要 1.5 µs（传输 + 延迟）+ 0.6 µs MMA，完全串行，结束在 18.3，`overlap = 0`。stages = 2 是 10.9，3 和 4 都是 10.6：再多 stage 也快不过 TMA 带宽。
:::

## 易错点

- **producer 从 phase 0 开始。** empty barrier 一开始没完成过任何 phase，`try_wait(0)` 不通过，三个 agent 全部卡住。测试 `test_pipeline_start_states`；完整运行时 Sim 报 `Deadlock`。
- **empty 只要 1 次 arrival。** 第一个 WG 做完就放行 producer，第二个 WG 还在读的 stage 被覆盖。`run_gemm` 的构造要求是 `Barrier(NUM_CONSUMERS)`；Sim 的 WAR 检查会抓住。
- **producer 跳过 empty wait。** TMA 写进一个还没被读完的 stage：Sim 报 `RaceError`（所有 stages 的正确性用例都会失败）。
- **expect_tx 只声明 A 的字节。** full 在 A 落地时就完成，consumer 读 B 时 B 还没到：Sim 报 `RAW on stage …`。PDL 路径里也一样：先发 B、后发 A，但声明的必须是 A+B。测试 `test_producer_pdl_prefetches_weights_before_grid_wait`。
- **越界部分不搬或少计字节。** TMA 的 box 是整块：边界 tile 里越界行必须是 0（否则累加了上一个 tile 的残留），字节数必须是整块，否则 tx 计数永远不归零。测试 `test_tma_box_load_swizzles_and_zero_fills`、`test_run_gemm_is_correct[MNK2-2]`。
- **trigger 放在 epilogue 之后或者两个 WG 都 trigger。** 测试 `test_consumer_op_sequence_and_single_trigger` 要求只有 WG 0、在最后一个 k-tile 之后、epilogue 之前。
- **PDL 后继在 grid_wait 之前读了 A。** A 是前驱的输出 H。测试检查 `grid_wait` 之前的 TMA 只有 B。

## 对照文档

> “多 stage pipeline 会用多个 shared-memory buffer：计算当前 stage 时，producer 预取后续 stage；consumer 在确认数据 ready 后再消费；stage 被用完后才能释放并复用。……例如一个 warp group 负责 TMA load（producer），其他 warp group 负责 MMA（consumer）。……这里增加的不只是新的 Layout：还需要角色分配、pipeline 状态、barrier、warp-group 同步以及架构特定的指令语义。”

文档伪代码和本题、CUTLASS 的对应关系：

| 文档伪代码 | 本题 | CUTLASS |
|---|---|---|
| `gA_tile = local_tile(gA, tile_MK, cta_coord)` | `run_gemm` 里的 `m0, n0` 和 `tma_box_load(src, r0, c0, …)` | `local_tile(mA_mkl, …)` + `tma_partition` |
| `sA = make_tensor(smem, swizzled_layout_A)` | `smemA` + `sw128` | `GMMA::Layout_K_SW128_Atom<half_t>` + `tile_to_shape` |
| `issue_load / wait_until_ready / release` | producer / consumer 的 wait、expect、tma、arrive | `producer_acquire` / `consumer_wait` / `consumer_release` |
| `gemm(tiled_mma, …)` | consumer 里 `acc[wg] += A_part · B_partᵀ` | `cute::gemm` + `warpgroup_arrive/commit/wait` |
| `store(gC_tile, accumulator)` | 最后一个 `compute(EPI_US, …)` | `collective_epilogue.store` |
| 后继 kernel 的 PDL | `chain(..., pdl=True)` | `launch_dependent_grids` / `wait_on_dependent_grids` |

## 上机验证

```bash
playground/run.sh playground/e1_full_gemm.cpp      # 真实 CuTe，本机编译运行
```

本机的真实输出：

```text
SW128 atom  : Sw<3,4,3> o smem_ptr[16b](unset) o (_8,_64):(_64,_1)
sA (128x64) : Sw<3,4,3> o smem_ptr[16b](unset) o ((_8,_16),(_64,_1)):((_64,_512),(_1,_0))
offset(0,8) = 8 offset(1,0) = 72 offset(1,8) = 64 offset(7,0) = 504 offset(8,0) = 512
sw128(m*64+k) vs real CuTe offset: 0 mismatches of 8192
stage bytes (A 128x64 + B 128x64, half) = 32768
```

两点值得注意。第一，CuTe 打印的是 `Sw<3,4,3>`，不是本题的 `Swizzle<3,3,3>`：真实的 swizzle 挂在 smem_ptr 上、作用在字节地址上（16 字节块 = 低 4 位），换算成 16-bit 元素偏移正好是 `o ^ ((o >> 3) & 0x38)`。只看 layout 的整数值会得到没 swizzle 的 64，必须通过真实指针算偏移才是 72，8192 个位置和给定的 `sw128` 全部一致。第二，一个 stage 的字节数 32768 就是 producer 传给 `arrive_and_expect_tx` 的数（`TmaTransactionBytes`）。

这个 GEMM 的真实版本需要 Hopper（SM90）的 WGMMA 和 TMA。RTX 5090（SM120）有 TMA 和 mbarrier，但没有 WGMMA，CUTLASS 在它上面用另一套 kernel；本题的协议（stage、phase、full / empty、warp specialization、PDL）在那里同样适用。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
st = PipelineState.producer_start(3)
# 连续 advance 7 次看到的 (index, phase)：
# [(0, 1), (1, 1), (2, 1), (0, 0), (1, 0), (2, 0), (0, 1)]
A = np.random.default_rng(1).standard_normal((128, 512)); B = ...       # 同形
C, r = run_gemm(A, B, Cfg(128, 128, 512, stages=1))
r   # {'end': 18.3, 'trigger': 17.3, 'tma_busy': 8.0, 'mma_busy': 4.8, 'overlap': 0.0}
C, r = run_gemm(A, B, Cfg(128, 128, 512, stages=3))
r   # {'end': 10.6, 'trigger': 9.6, 'tma_busy': 8.0, 'mma_busy': 4.8, 'overlap': 4.1}
chain(X, W1, W2, Cfg(128, 256, 256), Cfg(128, 128, 256), False)[1]   # [(0.0, 6.6), (9.6, 16.2)]
chain(X, W1, W2, Cfg(128, 256, 256), Cfg(128, 128, 256), True)[1]    # [(0.0, 6.6), (6.1, 13.0)]
```

#### 关键问题

1. producer 为什么从 phase 1 开始？如果从 0 开始会发生什么？
2. `empty[s]` 每个 phase 为什么要 2 次 arrival？`full[s]` 为什么只要 1 次？
3. 为什么“声明字节”和“arrive”要合成一个 `arrive_expect_tx`？如果先 arrive 再单独 expect_tx 会怎样？
4. PDL 路径里，为什么预取窗口是 `min(stages, k_tiles)` 个 stage？能不能把所有 k-tile 的 B 都在 `grid_wait` 之前发出去？
:::

::: hint 2 思路拆解与常见坑
- `PipelineState`：照着 CUTLASS 的 `operator++` 写，三个字段，两个 classmethod。
- `tma_box_load`：用 `np.arange` 做行、列索引，`inb = (r0+r < R) & (c0+c < C)`；先建全 0 的 vals，把在界内的元素填进去，再一次性写到 `smem[base + sw128(r*cols + c)]`。返回 `rows*cols*ELEM_BYTES`。
- 写一个小工具，给定 (stage, k-tile, "A" 或 "B") 返回一个闭包：调用时执行对应的 `tma_box_load`。stage 的基址是 `s*BM*BK`（A）或 `s*BN*BK`（B）。
- `producer`：先 yield setup，然后用 `producer_start`。PDL 时先做一个预取循环（wait empty → expect A+B → 只发 B → advance），`yield ("grid_wait",)` 之后用一个新的 `producer_start` 状态补发这些 stage 的 A，再从第 `min(stages, k_tiles)` 个 k-tile 开始正常循环。
- `consumer`：每个 k-tile 的 fn 用 `smem_read` 读出 A 的本 WG 行和整块 B，`acc[wg] += a @ b.T`；最后 WG 0 trigger，再 epilogue 写回（`C[m0+r, n0:n0+cols] = acc[...]`，裁掉越界部分）。
- `run_gemm`：每个 CTA 一个 Ctx 和一个 Sim；report 的 busy / overlap 用 CTA 0 的区间算。
- `chain`：第一个 GEMM 正常跑；第二个按 pdl 与否算 start 和 grid_ready。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. 一开始所有 stage 都是空的，producer 要等的是“上一轮已经被吃完”，而 empty barrier 还没完成过任何 phase：完成数 0 是偶数，`try_wait(1)` 立刻通过。从 0 开始，`try_wait(0)` 要等第一个 phase 完成，可 consumer 还在等 full，三方互等：死锁。
2. 两个 consumer WG 都读同一个 stage（各读自己那一半 A、整块 B），只有两个都读完才能覆盖，所以 empty 每轮要 2 次 arrival。full 的“到达”由 producer 一次 `arrive_expect_tx` 加上 TMA 的字节完成，所以 1 次 arrival 就够。
3. 分开写时，arrive 之后、expect_tx 之前的那一瞬间，barrier 看到“arrival 到齐、tx = 0”，phase 立刻完成，consumer 会在数据还没到时通过 wait。合成一条保证“声明字节”先于“到达”生效。
4. 每个 stage 只能装一个 k-tile 的数据；预取第 `stages+1` 个 k-tile 的 B 需要先复用 stage 0，而复用要等 consumer 读完 stage 0，consumer 读 stage 0 又要等 A，A 要等 `grid_wait`：所以 grid_wait 之前最多预取 `min(stages, k_tiles)` 个 stage。
:::
:::

## 参考解

::: solution solutions/e1_full_gemm.py
{{code: solutions/e1_full_gemm.py}}
:::

## 面试官追问 & 对照源码

- CUTLASS 的 consumer 不是 MMA 一结束就 release，而是 `warpgroup_wait<K_PIPE_MMAS>()` 之后才 release 上一个 stage（[[cutlass:include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp@mma|collective mma]]）。WGMMA 是异步的：本题的模型假设 MMA 同步完成，加入异步 MMA 后 empty 的时间线会怎样变？
- 真实的 epilogue 先把累加器写进 SMEM，再用 TMA store 写回（还要和下一个 tile 的 mainloop 共享 SMEM）。本题把它简化成一个 `compute(EPI_US)`：它在 Ping-pong 调度里为什么重要（D3）？
- 本题的 TMA 引擎是“每个 SM 一个、串行传输”。cluster multicast 时，一个 box 同时落到几个 CTA 的 SMEM，`expect_tx` 要怎样改？
- 两个 GEMM 串联时，`chain` 的第二个 kernel 在 PDL 下比不用 PDL 早结束 3.2 µs。把 `STREAM_GAP_US` 换成 CUDA Graph 的 1 µs 再算一次，PDL 还剩多少收益？这说明了什么（D4）？
