lede: 两个串联的 GEMM（H = X·W1，Y = H·W2）之间有一段只因为“前一个 kernel 还没结束”而空着的时间。PDL 让第二个 kernel 提前启动、先做和 H 无关的事，读 H 之前再等。本题把这条时间线算出来，并写一个检查器：trigger 放哪只影响快慢，wait 放哪决定对错。
# D4 · PDL：两个串联 GEMM 的提前启动、等待位置与正确性检查

## 题面速览

- **背景知识**：先读 [时间维度](../knowledge/time_dimension.html) 里的 PDL 一节。
- **在体系里的位置**：CUTLASS 把 PDL 的两条 device 端指令包成 [[cutlass:include/cutlass/arch/grid_dependency_control.h@launch_dependent_grids|launch_dependent_grids]] / [[cutlass:include/cutlass/arch/grid_dependency_control.h@wait_on_dependent_grids|wait_on_dependent_grids]]。Hopper 的 warp-specialized kernel 在 producer warp 发第一个 TMA 之前调用 wait（[[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp#L605|cooperative kernel 的 mainloop producer]]），在最后一个 tile 算完之后调用 launch（[[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp#L830|同一文件]]）。主机端用 `cudaLaunchAttributeProgrammaticStreamSerialization` 发射（[[cutlass:media/docs/cpp/dependent_kernel_launch.md|dependent_kernel_launch.md]]）。
- **你要实现**：`kernel_time`、`trigger_time`、`timeline`、`best_trigger`、`check`、`independent_operands`、`latest_wait`、`speedup`。`Phase`、`Kernel`、四个时间常量和例子构造函数 `two_gemm_chain` 已经给出。
- **关键规则**：
  1. 普通 stream / graph：后继 kernel 在前驱结束之后加一段固定空隙才开始；trigger 和 wait 都不起作用。
  2. PDL：后继在前驱的 trigger 之后 `PDL_LAUNCH_US` 开始；前驱没有 trigger 时退化成普通串行。
  3. `griddepcontrol.wait` 返回的时刻是**前驱完全结束 + FLUSH_US**，而不是前驱 trigger 的时刻。
  4. 正确性只看 wait：wait 之前不能读前驱写的 buffer，也不能写前驱读写的 buffer。trigger 放哪都不改变 `check` 的结果。
  5. 提前启动的后继会和前驱抢 SM：`contention` 让前驱的结束时间按重叠时长变长，所以 trigger 不是越早越好。

**验收要点**：kernel 与 trigger 时长 · stream / graph 的空隙 · stream 忽略 trigger 和 wait · PDL 的起点 22.5 与 wait 不卡（总时间 47.5）· wait 放在 W2 预取之前会卡到 26.5（49.5）· wait 等前驱结束而不是 trigger · 没有 trigger 时退化 · wait 在所有 phase 之后 · contention 让最优 trigger 从 0 变成 1 · 没有 PDL 时 contention 不起作用 · read-before-wait · write-before-wait（WAR）· 三个 kernel 的链只检查相邻对 · trigger 位置不改变检查结果 · `independent_operands = ['W2']`、`latest_wait = 2` · speedup · 未知 mode 报错。

::: callout info 第一次出现的术语
- **PDL**（Programmatic Dependent Launch）：Hopper（sm90）起的 CUDA 特性。同一个 stream 里的后继 kernel 可以在前驱结束前启动。
- **trigger**：前驱调用 `griddepcontrol.launch_dependents`（CUDA runtime 里是 `cudaTriggerProgrammaticLaunchCompletion()`），表示“后继可以开始了”。它只是性能提示，不提供可见性。
- **wait**：后继调用 `griddepcontrol.wait`（`cudaGridDependencySynchronize()`），阻塞到前驱全部结束、写回的内存对它可见。
- **独立 operand**：后继读、但前驱不写的数据，例如第二个 GEMM 的权重 W2。它可以在 wait 之前加载。
- **WAR**（write-after-read）：后继写了前驱还要读的 buffer。前驱还在跑，提前写会破坏它的输入。
:::

## 直觉与类比

::: callout analogy 接力赛的交接区
第一棒（gemm1）跑到交接区入口时喊一声“我到了”（trigger），第二棒（gemm2）听到就开始起跑加速（setup、预取 W2）。但接力棒（H）必须真正交到手上才能继续跑（wait）：起跑早不等于可以不接棒就跑。喊得太早，第二棒会在跑道上挡住第一棒（contention）；喊得太晚，交接区里就是空等。没有 PDL 的普通 stream 相当于第二棒站在原地，等第一棒完全停下来、再等裁判吹哨（3 µs 的 launch 空隙）才起跑。
:::

## PDL 的价值在哪一层体现

单独测一个 GEMM，PDL 几乎看不出收益：它节省的是**两个 kernel 之间**的空隙（launch 延迟、prologue、和前驱无关的预取），只有在一串依赖的 kernel 里才有意义。所以它真正发挥作用的地方在系统层：

- **CUDA Graph / stream 里的一串小 kernel**：LLM decode 阶段每层是一串小 GEMM 和逐元素 kernel，每个只跑几微秒，launch 和 setup 的固定开销占比很高。graph 只能把空隙从 ~3 µs 压到 ~1 µs，PDL 能把下一个 kernel 的 setup 和权重预取藏进上一个的尾巴里。
- **编译器 / 框架负责插 trigger 和 wait**：哪个 kernel 读哪个 buffer、wait 最晚能放到哪里（本题的 `latest_wait`）、trigger 该放多早（`best_trigger`）都是**跨 kernel** 的信息，单个 kernel 的作者看不全。CUTLASS 把指令包好，但“这一对 kernel 之间能不能用、用在哪”通常由图级调度（CUDA Graph 的 programmatic edge、推理框架的 kernel 编排）决定。
- **autotune 要按端到端时间做**：CUTLASS example 63 的 README 也提醒，大部分收益要在端到端应用里看，`overlap_ratio` 和 `prefetch_ratio` 应该按整条链的时间调。本题的 `best_trigger` 是这个搜索的最小模型。

## 图解：四种 mode 的时间关系

::: flow gemm1 → gemm2 的起点与等待
cfg cw=210 rh=100 nw=176
n g1 0 0.5 "gemm1" "2 + 20 + 4 = 26 µs" k=acc
n trg 1 0 "trigger" "mainloop 之后：22 µs" k=warn
n end 1 1 "前驱结束" "26 µs（+0.5 flush）" k=dark
n st 2 0 "gemm2 启动" "PDL：22 + 0.5" k=teal
n w 2 1 "wait 返回" "max(t, 26.5)" k=todo
n ml 3 0.5 "读 H 的 mainloop" "wait 之后" k=pass
e g1 -> trg
e g1 -> end
e trg -> st
e end -> w
e st -> ml
e w -> ml
cap: 起点只由 trigger 决定，wait 只由前驱结束决定。正确性来自下面那条线；trigger 那条线只决定能藏住多少 setup 和预取。
:::

## 逐步演示

::: steps 文档里的 H = X·W1 → Y = H·W2：从 stream 54 µs 到 PDL 44.5 µs
::: flow
cfg cw=200 rh=80 nw=168
n s 0 0 "stream" "54.0" k=dark
n g 1 0 "graph" "52.0" k=acc
n p 2 0 "PDL" "47.5" k=pass
n c 3 0 "contention" "最优 trigger 变化" k=warn
n e 2 1 "wait 太早" "49.5" k=todo
n k 3 1 "check" "read-before-wait" k=todo
e s -> g
e g -> p
e p -> c
e p -> e dash
e e -> k dash
:::
--- step 普通 stream {hl=s}
gemm1 = 2 + 20 + 4 = 26 µs，gemm2 = 2 + 5 + 15 + 3 = 25 µs。stream 模式下 gemm2 从 26 + 3 = 29 开始，spans `[(0, 26), (29, 54)]`，总时间 54。
--- step CUDA graph {hl=s,g}
graph 节点之间的空隙是 1 µs：gemm2 从 27 开始，总时间 52。graph 只缩短 launch 空隙，不让两个 kernel 重叠。
--- step PDL：提前启动 {hl=g,p}
gemm1 在 mainloop 之后 trigger（22 µs），gemm2 从 22 + 0.5 = 22.5 开始：setup 22.5–24.5，prefetch_w2 24.5–29.5。wait 放在 mainloop 之前，返回时刻是 `max(29.5, 26 + 0.5) = 29.5`，没有卡住。mainloop 29.5–44.5，epilogue 到 47.5。`speedup = 54 / 47.5 = 1.137`。
--- step wait 放得太早 {hl=p,e}
`two_gemm_chain(consumer_wait="prefetch_w2")`：wait 在 W2 预取之前，prefetch 只能从 26.5 开始，总时间 49.5。这个位置是正确的，但把能藏在 gemm1 后面的 5 µs 预取浪费了。`latest_wait` 返回 2（mainloop），这是正确位置里最晚、也最快的。
--- step contention 改变最优 trigger {hl=p,c}
没有 contention 时 `best_trigger = (0, 44.5)`：setup 之后就 trigger 最好。contention = 0.5 时，`trigger_after = 0` 让 gemm2 的 setup + prefetch（7 µs）和 gemm1 重叠，gemm1 被拉长到 `26 + 0.5·7 = 29.5`，总时间 48.0；`trigger_after = 1` 时 gemm2 从 22.5 开始，只和 gemm1 重叠 26 − 22.5 = 3.5 µs，gemm1 拉长到 27.75，wait 在 28.25 返回，仍早于 mainloop 的 29.5，总时间不变，最优变成 `(1, 47.5)`。这就是 CUTLASS 注释说的“太早会和当前 kernel 抢资源”。
--- step 正确性检查 {hl=e,k}
去掉 wait（`consumer_wait="none"`），`check` 返回 `["read-before-wait: gemm2.mainloop reads H"]`。如果 gemm2 的 setup 写了 X（gemm1 还在读），返回 `["write-before-wait: gemm2.setup writes X"]`。改 trigger 的位置，这两条结果都不会变。
:::

## 易错点

- **wait 等到前驱的 trigger 就返回。** trigger 不提供可见性，wait 等的是前驱完全结束 + flush。测试 `test_wait_waits_for_end_not_trigger` 把 epilogue 拉长到 30 µs，mainloop 必须从 52.5 开始。
- **前驱没有 trigger 时后继在前驱开始时启动。** 没有 trigger 等价于“kernel 结束时隐式触发”，PDL 退化成串行加空隙。测试 `test_no_trigger_falls_back_to_gap`。
- **stream 模式也用了 trigger。** 只有 pdl / graph_pdl 读 trigger 和 wait。测试 `test_timeline_modes`。
- **contention 拉长了前驱的 trigger。** 只拉长前驱的结束，trigger 时刻不变（后继已经启动了）。测试 `test_contention_only_stretches_the_producer_end` 检查 spans 是 `(0, 27.75)`。
- **只检查读。** 后继在 wait 之前写前驱的输入（WAR）同样是竞争。测试 `test_write_hazard_before_wait`。
- **用 trigger 的位置推断正确性。** 测试 `test_trigger_position_never_changes_correctness` 遍历所有组合。

## 对照文档

> “PDL 不等于 kernel 内 pipeline。它解决的是两个有依赖关系的 kernel 之间的启动时机：允许 consumer kernel 在 producer kernel 完全结束之前就开始执行一部分与 producer 输出无关的工作。……Consumer 只能在同步前访问明确独立于 producer 的数据。不能因为 kernel 已经提前启动，就提前读取 producer 尚未完成写入的输出。”

文档里的“Producer 先完成必要写入 → 触发 programmatic completion → Consumer 先做不依赖 producer 的工作 → grid dependency synchronize → 再读取依赖 producer 的结果”五个框，在本题里分别是 `trigger_after`、`PDL_LAUNCH_US`、`prefetch_w2`、`wait_before`、`mainloop`。文档说“预取 W₂、准备地址”可以提前做，`independent_operands` 就是把这句话变成集合运算：consumer 的 reads 减去 producer 的 writes。

## 真实代码

两条指令和它们的注释（简化自 [[cutlass:include/cutlass/arch/grid_dependency_control.h|grid_dependency_control.h]]）：

```cpp
// launch_dependents doesn't impact the functionality but the performance:
// Launching a dependent kernel too early can compete with current kernels,
// while launching too late can lead to a long latency.
void launch_dependent_grids() { asm volatile("griddepcontrol.launch_dependents;"); }
// enforces no global memory access prior to this instruction.
void wait_on_dependent_grids() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
```

example 63 的 producer（简化自 [[cutlass:examples/63_hopper_gemm_with_weight_prefetch/collective/sm90_mma_tma_gmma_ss_warpspecialized_with_prefetch.hpp#L423-L460|sm90_mma_tma_gmma_ss_warpspecialized_with_prefetch.hpp]]）：加载激活（前驱的输出）的 warp 先 wait，加载权重的 warp 不等；trigger 在发出一定比例的 k-tile 之后：

```cpp
cutlass::arch::wait_on_dependent_grids();
for (int cnt = 0; k_tile_count > 0; --k_tile_count, ++cnt) {
  pipeline.producer_acquire(smem_pipe_write);
  copy(tma_load_a ...); copy(tma_load_b ...);
  if (!disable_gdc && cnt >= launch_dep_grids_threshold && !launch_dep_grids) {   // overlap_ratio 决定阈值
    launch_dep_grids = true;
    cutlass::arch::launch_dependent_grids();
  }
  ++smem_pipe_write;
}
```

| 文档 / 本题 | CUTLASS |
|---|---|
| `trigger_after` | `overlap_ratio` 决定的 `launch_dep_grids_threshold` |
| `wait_before` | producer warp 里 `wait_on_dependent_grids()` 的位置 |
| `independent_operands` | example 63 在 wait 之前预取的权重（`prefetch_ratio`） |
| `contention` | 注释里的 “compete with current kernels” |

## 上机验证

本机没有 Hopper GPU，PDL 的真实效果没法在这里测。有 sm90（或更新）的机器时可以这样对照：

```bash
# CUTLASS example 63：overlap_ratio 就是本题的 trigger 位置，-1 表示关闭 PDL 重叠
./63_hopper_gemm_with_weight_prefetch --m=8192 --n=1 --k=8192 --o=-1.0 --p=-1.0
./63_hopper_gemm_with_weight_prefetch --m=8192 --n=1 --k=8192 --o=0.5 --p=-1.0
./63_hopper_gemm_with_weight_prefetch --m=8192 --n=1 --k=8192 --o=0.8 --p=0.7
# 用 Nsight Systems 看两个 kernel 是否重叠：
nsys profile --cuda-graph-trace=node -o pdl ./your_two_gemm_app
```

RTX 5090（SM120）也支持 `griddepcontrol`，[cuda-kernels-dev-drills](https://github.com/kzhou003/cuda-kernels-dev-drills) 里有在 5090 上真机测 PDL 的练习。

## 三级提示

::: hint 1 I/O 例子和关键问题
```python
ks = two_gemm_chain()
kernel_time(ks[0]), trigger_time(ks[0])          # (26.0, 22.0)
timeline(ks, "stream")["spans"]                  # [(0.0, 26.0), (29.0, 54.0)]
timeline(ks, "pdl")["phases"][1]
# [('setup', 22.5, 24.5), ('prefetch_w2', 24.5, 29.5), ('mainloop', 29.5, 44.5), ('epilogue', 44.5, 47.5)]
best_trigger(ks, "pdl", 0.0), best_trigger(ks, "pdl", 0.5)   # ((0, 44.5), (1, 47.5))
check(two_gemm_chain(consumer_wait="none"))      # ['read-before-wait: gemm2.mainloop reads H']
independent_operands(*ks), latest_wait(*ks)      # (['W2'], 2)
```

#### 关键问题

1. 后继 kernel 的起点由前驱的哪个时刻决定？wait 返回的时刻呢？
2. 为什么 trigger 的位置不影响正确性，而 wait 的位置影响？
3. contention 存在时，为什么最早的 trigger 不一定最好？
4. wait 能不能放得比“第一个读 H 的 phase”更晚？更早会怎样？
:::

::: hint 2 思路拆解与常见坑
- `timeline` 按 kernel 顺序走，维护“前驱的 start、trigger 时刻、实际结束时间”。kernel 0 从 0 开始。对 kernel i ≥ 1，先算起点（规则 1、2），然后逐个 phase 推进时间 t；到第 `wait_before` 个 phase 时 `t = max(t, prev_end + FLUSH_US)`。`wait_before == len(phases)` 时在循环之后对结束时间做同样的 max。
- contention：先用“没有 wait 的名义时间”算出后继第一个等待 phase 的开始 w（没有 wait 时是后继跑完的时间），`o = max(0, min(E0, w) − start)`，把前驱的结束改成 `E0 + contention·o`，再用新的结束时间算 wait。别忘了更新 spans 里前驱那一项。
- `check`：对每对相邻 kernel，求前驱的 reads ∪ writes 和 writes；遍历后继 wait 之前的 phase（没有 wait 时是全部），先报读冲突再报写冲突，buffer 名字排序。
- `best_trigger`：用 `dataclasses.replace` 改 `trigger_after`，比较时用 `<=`，这样并列时留下更晚的 trigger。
:::

::: hint 3 关键问题的答案
::: answer 答案
1. 起点由前驱的 trigger 决定（trigger 时刻 + `PDL_LAUNCH_US`），前驱没有 trigger 时由前驱结束决定；wait 返回的时刻永远由前驱**结束**决定（+ flush），和 trigger 无关。
2. trigger 只是“可以开始调度后继”的提示，不带任何内存语义；wait 才是唯一的同步点，它保证前驱的所有写对后继可见。所以数据依赖只能由 wait 的位置保证。
3. 后继提前启动后，它的 CTA 要占 SM、寄存器、SMEM 和带宽。如果前驱还有很多工作没做完，这些资源被抢走会拉长前驱，而前驱的结束又决定了 wait 返回的时刻：trigger 太早可能得不偿失。
4. 可以更晚（例如 epilogue 前），但没有必要，只会白白推迟；不能更早于第一个读 H 的 phase，否则就是 read-before-wait。最晚的**正确**位置 `latest_wait` 通常也是最快的位置：它让最多的独立工作藏在前驱后面。
:::
:::

## 参考解

::: solution solutions/d4_pdl_chain.py
{{code: solutions/d4_pdl_chain.py}}
:::

## 面试官追问 & 对照源码

- 三个 kernel 的链 A → B → C：C 能不能在 A 结束前启动？`griddepcontrol.wait` 等的是“直接前驱”还是“所有前驱”？
- CUDA Graph 里怎样表达 PDL 的依赖边（programmatic dependency edge）？它和普通边在 graph 实例化时有什么不同？
- 如果后继 kernel 的 CTA 数比 SM 数还多，提前启动的那部分 CTA 会怎样占资源？这和本题的 contention 模型有什么出入？
- 推理框架里，哪些相邻 kernel 对最值得开 PDL（提示：短 kernel、有大的独立 operand、前驱尾部有大块 epilogue）？
