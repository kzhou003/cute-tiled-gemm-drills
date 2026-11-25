lede: 有了 TMA 和 mbarrier 流水线，发搬运指令的线程几乎不干活，却和做 MMA 的线程占着一样多的寄存器。Hopper 的 CUTLASS GEMM 把一个 CTA 拆成 1 个 producer warp group 和 2 个 consumer warp group，用 setmaxnreg 把寄存器从前者挪给后者；两个 consumer 再按 Cooperative 或 Ping-pong 两种方式分工。本题算清这笔寄存器账（为什么恰好是 40 和 232），并模拟两种调度的时间线，找出 Ping-pong 什么时候赢、什么时候反而输。
# D3 · Warp specialization：setmaxnreg 寄存器预算与 Cooperative / Ping-pong 调度

## 题面速览

- **背景知识**：先读 [时间维度](../knowledge/time_dimension.html) 的 warp specialization 一节；D2 的流水线是这里 producer 和 consumer 之间的通道。
- **在体系里的位置**：两份 kernel 是 [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp|cooperative]] 和 [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp|pingpong]]。寄存器常数在 [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp@LoadRegisterRequirement|pingpong 的 LoadRegisterRequirement]]，重分配指令在 [[cutlass:include/cutlass/arch/reg_reconfig.h|reg_reconfig.h]]，Ping-pong 的有序屏障是 [[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@OrderedSequenceBarrier|OrderedSequenceBarrier]]。
- **你要实现**：`launch_regs_per_thread`、`validate_setmaxnreg`、`register_plan`、`accumulator_regs`、`schedule_cooperative`、`schedule_pingpong`、`steady_per_tile`。常数和 `Segment`、`Schedule` 已经给出。
- **关键规则**：
  1. 启动时每线程寄存器 = `floor(65536 / threads / 8) × 8`；384 线程是 168。
  2. `setmaxnreg` 的目标值是 8 的倍数、在 [24, 256]；dec 只能降、inc 只能升；重分配后的总数不能超过启动时的总数。
  3. Cooperative：两个 WG 一起算一个 tile，每 tile `M + E/2`。
  4. Ping-pong：tile j 归 WG `1 + j%2`；MMA 链和 epilogue 链各自严格交替；每 tile `max(M, E)`。

**验收要点**：启动寄存器按 8 取整 · `setmaxnreg` 的两条规则 · CUTLASS 默认的 40 / 232 恰好用满 64512 · 重负载的 24 / 240 也恰好用满 · 两种混用超出 2048 · 方向错误和对齐错误 · 3 个 consumer WG 时最多只能升到 160 · 累加器寄存器（64、128）· Ping-pong 4 个 tile 与 Cooperative 2 个 tile 的逐段时间线 · 三组参数下 8 个 tile 的 makespan · Tensor Core 利用率 · Ping-pong 的交替与不自重叠 · 稳态公式与模拟一致 · E 超过 2M 时 Cooperative 反超 · 空调度。

::: callout info 第一次出现的术语
- **WG**（warp group）：4 个连续 warp = 128 线程。WGMMA（warp-group 级矩阵乘指令）以 WG 为单位发射。
- **warp specialization**：同一个 CTA 里不同 WG 执行不同的代码路径（producer 只发 TMA，consumer 只做 MMA 和 epilogue），用 mbarrier 通信。
- **setmaxnreg**（`setmaxnreg.inc/dec.sync.aligned.u32 N`）：Hopper 的 PTX 指令，整个 WG 一起把每线程寄存器上限改成 N。dec 释放的寄存器回到 CTA 的池子，inc 从池子里取。
- **epilogue**：MMA 算完之后把累加器转换、加 bias、写回 C 的那一段。Hopper 上通常经 SMEM 再用 TMA store。
- **persistent kernel**：CTA 数等于 SM 数，每个 CTA 循环处理多个输出 tile。两个 consumer 交替处理 tile 就建立在这上面。
:::

## 直觉与类比

::: callout analogy 一间厨房三个人
一间厨房只有一台大灶（Tensor Core），三个人分工：一个跑腿买菜（producer：发 TMA），两个掌勺（consumer）。跑腿只需要一个小篮子，所以把他的大推车（寄存器）让给两个掌勺的放配料——这就是 `setmaxnreg`：总推车数不变，40 + 232 + 232 正好等于开门时每人 168 的总和。两个掌勺怎么用一台灶？Cooperative 是两人一起炒同一锅大菜，炒完再一起装盘，装盘时灶空着；Ping-pong 是两人各炒一锅，A 装盘的时候 B 正在灶上炒，灶几乎不停。但装盘台（epilogue 的 SMEM 和 TMA store）也只有一个，两人轮流用：如果装盘比炒菜还慢，瓶颈就变成装盘台。
:::

## 图解：一个 CTA 的三个 WG

::: flow 384 线程的分工
cfg cw=190 rh=100 nw=168
n p 0 0.5 "WG0 Producer" "setmaxnreg.dec 40" k=acc
n pipe 1 0.5 "mbarrier 流水线" "full / empty × S" k=teal
n c1 2 0 "WG1 Consumer0" "setmaxnreg.inc 232" k=pass
n c2 2 1 "WG2 Consumer1" "setmaxnreg.inc 232" k=pass
n tc 3 0.5 "Tensor Core" "MMA：一次一个 WG" k=vio
n epi 4 0.5 "Epilogue" "写回：一次一个 WG" k=dark
e p -> pipe
e pipe -> c1
e pipe -> c2
e c1 -> tc
e c2 -> tc
e tc -> epi
cap: 寄存器账：128 × (40 + 232 + 232) = 64512 = 168 × 384。Ping-pong 里 WG1 和 WG2 交替占用 Tensor Core 和 epilogue；Cooperative 里两者同时做同一件事。
:::

## 逐步演示

::: steps M = 10，E = 4 的 Ping-pong，4 个 tile
::: flow
cfg cw=200 rh=80 nw=170
n t0 0 0 "WG1 · tile 0" "MMA 0–10" k=pass
n t1 1 0 "WG2 · tile 1" "MMA 10–20" k=teal
n t2 2 0 "WG1 · tile 2" "MMA 20–30" k=pass
n st 3 0 "稳态" "每 tile 10" k=vio
e t0 -> t1
e t1 -> t2
e t2 -> st
:::
--- step t = 0–10：WG1 算 tile 0 {hl=t0}
`OrderedSequenceBarrier` 的 group 0 从相反的 phase 起步，所以 WG1 的第一次 wait 立即通过；WG2 的 wait 要等 WG1 arrive。段：`(1, mma, 0, 0, 10)`。
--- step t = 10–20：WG2 算 tile 1，WG1 写回 tile 0 {hl=t0,t1}
WG1 在 10 时 arrive（把 MMA 位交给 WG2），然后做 epilogue `(1, epi, 0, 10, 14)`；同时 WG2 `(2, mma, 1, 10, 20)`。Tensor Core 没有停。
--- step t = 20–30：WG1 回来算 tile 2 {hl=t1,t2}
tile 2 的 MMA 要同时等“tile 1 的 MMA 结束”（20）和“WG1 自己写完 tile 0”（14）：`max(20, 14) = 20`。WG2 在 20–24 写回 tile 1。
--- step 稳态 {hl=t2,st}
4 个 tile 的 makespan 44，Tensor Core 忙 40。8 个 tile 时 Ping-pong 84、Cooperative 96（每 tile 10 + 4/2 = 12），利用率 0.952 对 0.833。E = 10 时是 90 对 120；E = 16 时 epilogue 链成为瓶颈，每 tile 16，仍略胜 Cooperative 的 18（138 对 144）。
:::

## 易错点

- **以为 setmaxnreg 能凭空多出寄存器。** inc 只能拿走 dec 放回池子的部分。40 / 240 混用需要 18432 个却只放出 16384 个：`E_BUDGET: final 66560 registers > launch allocation 64512`。测试 `test_register_plan_errors`。
- **启动寄存器不按 8 取整。** 65536 / 384 = 170.67，实际只有 168；按 170 算账就对不上。测试 `test_launch_regs_round_down_to_8`。
- **inc 的目标低于启动值。** 测试 `test_register_plan_errors` 里 160 < 168 报 `E_ALLOC_BELOW_LAUNCH`。
- **以为 Cooperative 两个 WG 一起算就快一倍。** Tensor Core 是 SM 共享的；两个 WG 一起算只是把一个 tile 分成两半，MMA 时间不变。真正省下的是 epilogue 各写一半（E/2）。测试 `test_cooperative_timeline`。
- **把 Ping-pong 的稳态写成 (M + E) / 2。** 那只是单个 WG 的约束。epilogue 也被有序屏障串起来，所以还要不小于 E：稳态是 `max(M, E)`。测试 `test_cooperative_wins_when_epilogue_is_long`。
- **只让 MMA 交替、忘了同一个 WG 要先写完上一个 tile。** tile j 的 MMA 必须等 `epi_end[j−2]`：累加器寄存器还被上一个 tile 占着。测试 `test_pingpong_invariants_on_random_inputs` 检查一个 WG 的各段互不重叠。

## 对照文档

> “例如一个 warp group 负责 TMA load（producer），其他 warp group 负责 MMA（consumer）。这种分工可以减少搬运和计算之间的资源竞争，并让异步执行更清晰。这里增加的不只是新的 Layout：还需要角色分配、pipeline 状态、barrier、warp-group 同步以及架构特定的指令语义。”

文档说的“资源竞争”，最具体的一项就是寄存器：本题把它量化成 64512 这笔账。“warp-group 同步”在 Ping-pong 里是 MMA 和 epilogue 两条有序链；“架构特定的指令语义”是 `setmaxnreg` 的 8 对齐、[24, 256] 范围和 inc / dec 的方向。

## 真实代码

寄存器常数（简化自 [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp@LoadRegisterRequirement|pingpong kernel]]，Cooperative 的常数相同）：

```cpp
static_assert(NumMMAThreads == 128, "Pingpong kernel must have TiledMMA operating using 128 threads.");
static_assert(MaxThreadsPerBlock == 384, "Pingpong kernel must have 384 threads in total.");
static constexpr int RegsPerThread = (TileM * TileN * sizeof(ElementAccumulator)) / (NumMMAThreads * sizeof(uint32_t));
static constexpr bool HeavyRegisterPressure = RegsPerThread >= 208;
static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
static constexpr uint32_t MmaRegisterRequirement = !HeavyRegisterPressure ? 232 : 240;
```

`RegsPerThread` 就是本题的 `accumulator_regs`。角色分派和两条有序链（简化自 pingpong 的 `operator()`）：

```cpp
if (warp_group_role == WarpGroupRole::Producer) {
  cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();    // setmaxnreg.dec 40
  ...                                                                 // TMA 发射、调度器、epilogue load
} else {                                                              // Consumer0 / Consumer1
  cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();       // setmaxnreg.inc 232
  while (work_tile_info.is_valid()) {
    math_wg_order_barrier.wait();  collective_mainloop.mma(...);      math_wg_order_barrier.arrive();
    math_wg_order_barrier.wait();  collective_epilogue.store(...);    math_wg_order_barrier.arrive();
    ...                                                               // 跳到下一个属于自己的 tile（隔一个）
  }
}
```

有序屏障的关键：group 0 从相反的 phase 起步（“Group 0 starts with an opposite phase”），`arrive()` 通知 `(group_id + 1) % Length` 后 `++stage_`。`SequenceDepth = 2` 的两个 stage 正好是“MMA”和“epilogue”，所以两条链各自交替。

## 上机验证

这道题没有 C++ oracle：kernel 和有序屏障都离不开 GPU（`setmaxnreg`、mbarrier 是 PTX），CPU 上编译不出可运行的程序。所以期望值来自源码里的常数（上面引用的行）和本题的时间模型。

```bash
DRILL_IMPL=solutions python3 -c "from drill import load; m = load('d3_warp_specialization'); print(m.register_plan())"
grep -n "RegisterRequirement\|math_wg_order_barrier\.\|warpgroup_reg_" \
  third_party/cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp
```

有 Hopper（SM90）GPU 时，用 `cuobjdump -sass` 看 CUTLASS 的 warp-specialized kernel，应当能找到 `setmaxnreg` 对应的 SASS 指令，`cuobjdump -res-usage` 报告的寄存器数应当是 168；用 Nsight Compute 对比 `KernelTmaWarpSpecializedCooperative` 与 `KernelTmaWarpSpecializedPingpong` 的 Tensor Core 利用率，可以检验本题的时间模型（这两点本机没有 GPU，未核对）。RTX 5090 是 SM120，不支持 WGMMA；在它上面跑的是另一套 Blackwell 消费级的 kernel，`setmaxnreg` 的账也要按那套 kernel 重算。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
launch_regs_per_thread(384), launch_regs_per_thread(512)   # (168, 128)
register_plan(load_regs=40, mma_regs=240)["errors"]        # ['E_BUDGET: final 66560 registers > launch allocation 64512']
accumulator_regs(128, 128, 256), accumulator_regs(128, 128, 128)   # (64, 128)
s = schedule_pingpong(4, 10, 4)
[(x.wg, x.kind, x.tile, x.start, x.end) for x in s.segments][:3]
# [(1, 'mma', 0, 0.0, 10.0), (1, 'epi', 0, 10.0, 14.0), (2, 'mma', 1, 10.0, 20.0)]
schedule_cooperative(8, 10, 4).makespan, schedule_pingpong(8, 10, 4).makespan     # (96.0, 84.0)
steady_per_tile("pingpong", 10, 16), steady_per_tile("cooperative", 10, 16)      # (16, 18.0)
```

#### 关键问题

1. 40 和 232 是怎么来的？如果改成 3 个 consumer WG（512 线程），consumer 最多能升到多少？
2. 同样 128×128 的 tile，Cooperative 和 Ping-pong 每个 consumer 线程各要多少累加器寄存器？这对 tile 大小的选择意味着什么？
3. 在什么条件下 Ping-pong 比 Cooperative 快？什么时候反过来？
:::

::: hint 2 思路拆解与常见坑
- `register_plan`：算出 threads、launch、两个总数，然后按顺序检查：两个目标值各自合法 → `load_regs ≤ launch` → `mma_regs ≥ launch` → `final ≤ launch_total`。
- `schedule_cooperative`：一个时间游标 t；每个 tile 给 WG1、WG2 各记一段 `[t, t+M]` 的 mma，再各记一段 `[t+M, t+M+E/2]` 的 epi。
- `schedule_pingpong`：两个字典 `mma_end`、`epi_end`；按递推式逐个 tile 算，不存在的项按 0。
- 最后统一排序（开始时间，再 wg），makespan 取最大结束时刻，`tensor_core_busy = n·M`。
- `steady_per_tile`：写出公式，再用 `(makespan(n+2) − makespan(n)) / 2` 自己验证一遍。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. 384 线程启动时每线程 168，总 64512。producer 降到 40 放出 128 × 128 = 16384 个；两个 consumer 各升 64，正好要 2 × 128 × 64 = 16384。512 线程时启动 128，producer 降到 24 放出 13312 个，3 个 consumer 平分每线程最多再加 34.67，按 8 对齐只能到 160。
2. Cooperative 两个 WG 分一个 tile，每线程 128·128/256 = 64；Ping-pong 一个 WG 拿整个 tile，每线程 128。所以 Ping-pong 的 tile 通常要小一半（例如 64×128），或者接受更高的寄存器压力（≥ 208 时 CUTLASS 改用 24 / 240）。
3. Ping-pong 每 tile `max(M, E)`，Cooperative 每 tile `M + E/2`。`max(M, E) ≤ M + E/2` 当且仅当 `E ≤ 2M`：epilogue 不超过 MMA 的两倍时 Ping-pong 不慢，E ≤ M 时 epilogue 被完全藏住。E > 2M（很小的 K、很重的 epilogue fusion）时 Cooperative 更快。E = 0 时两者相同。
:::
:::

## 参考解

::: solution solutions/d3_warp_specialization.py
{{code: solutions/d3_warp_specialization.py}}
:::

## 面试官追问 & 对照源码

- Producer WG 的 4 个 warp 各干什么？对照 pingpong kernel 里的 `WarpGroupRole` / `ProducerWarpRole`：Mainloop 发 A/B 的 TMA，另一个 warp 跑 tile 调度器，Epilogue 发 C 的 TMA load。
- 本题假设 TMA 总能跟上。如果 D2 的流水线 stage 不够，Ping-pong 的两个 consumer 抢同一条流水线，会怎样？（提示：Ping-pong 的 `mainloop_pipe_consumer_state.advance(k_tile_count * NumMmaWarpGroups)` 跳过了另一个 WG 用掉的 stage。）
- Ping-pong kernel 在 `scheduler.is_last_tile(...)` 为真时调用 `launch_dependent_grids()`：这就是 PDL 的触发点，D4 会用到。
- Blackwell（SM100）把累加器放进 Tensor Memory，不再占寄存器：`setmaxnreg` 这笔账还需要算吗？
