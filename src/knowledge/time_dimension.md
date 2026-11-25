lede: Layout 解决“数据在哪里、由谁持有”；执行机制还要解决“何时搬、何时算、谁先运行、谁必须等待”。这一页讲 TMA、异步流水线、warp specialization 和 PDL 怎样加入一个 GEMM，以及它们和 Layout Algebra 的耦合程度，对应 Track D。
# 时间维度：TMA、Pipeline、Warp Specialization、PDL

## TMA：把大块搬运交给专用硬件

传统的线程协作 copy 要每个线程生成地址、发 load / store。TMA 让一个指定线程发起描述符驱动的多维搬运，由硬件完成整块 tensor 的传输（D1，[[cutlass:media/docs/cpp/cute/0z_tma_tensors.md|0z_tma_tensors.md]]）。三样东西各管一件事：

| 东西 | 管什么 |
|---|---|
| layout | tensor 的形状、stride、SMEM 排布和 swizzle |
| Copy Atom / TMA 描述符 | 用什么机制搬、搬哪个 tile（box），见 [[cutlass:include/cute/atom/copy_traits_sm90_tma.hpp@make_tma_copy_desc|make_tma_copy_desc]] |
| barrier / pipeline | 数据什么时候到、消费者什么时候能读 |

TMA 没有让 layout 失去意义：它改变的是“谁执行搬运”，不是“数据的坐标和物理排布”。

## 异步流水线：让搬运和计算重叠

顺序执行是 `load 0 → compute 0 → load 1 → compute 1`，搬运延迟完全暴露。多 stage 流水线用多个 SMEM buffer：计算当前 stage 时，producer 预取后面的 stage；consumer 确认数据 ready 后再读；用完后释放，producer 才能复用（D2，[[cutlass:media/docs/cpp/pipeline.md|pipeline.md]]）。

::: steps 一个 stage 的生命周期（CUTLASS PipelineTmaAsync）
::: flow
cfg cw=280 rh=96 nw=170
n acq 0 0 "producer_acquire" "等 empty[s]" k=acc
n tma 1 0 "TMA 发出" "expect_tx 字节" k=vio
n full 2 0 "full[s] 到齐" "phase 翻转" k=teal
n wait 2 1 "consumer_wait" "等 full[s]" k=warn
n mma 1 1 "WGMMA 计算" "读 SMEM" k=pass
n rel 0 1 "consumer_release" "arrive empty[s]" k=todo
e acq -> tma
e tma -> full "complete_tx" anim
e full -> wait
e wait -> mma
e mma -> rel
e rel -> acq "buffer 回收" dash
:::
--- step 拿到空 buffer {hl=acq,tma}
producer 在 `empty[s]` 上按当前 phase 等待，确认这个 stage 已经被 consumer 释放；然后 leader 线程在 `full[s]` 上 `arrive_and_expect_tx(transaction_bytes)`：告诉屏障“还会有这么多字节到达”（[[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@producer_acquire|producer_acquire]]）。
--- step 发出 TMA {hl=tma,full}
`copy(tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(...), tAsA(...))`：TMA 搬完会自动在 `full[s]` 上扣掉对应字节数（[[cutlass:include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp@load|collective load]]）。
--- step 屏障完成 {hl=full}
到达数和字节数都清零时，mbarrier 的 phase 位翻转。等待方看的就是这个 phase。
--- step consumer 等数据 {hl=full,wait}
consumer 在 `full[s]` 上按自己的 phase 等待（[[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@consumer_wait|consumer_wait]]）。phase 记错就会死锁，或者读到上一轮的数据：D2 的测试专门构造这些错误。
--- step 计算并释放 {hl=wait,mma,rel}
`warpgroup_arrive(); cute::gemm(...); warpgroup_commit_batch(); warpgroup_wait<K_PIPE_MMAS>();` 之后 `consumer_release` 在 `empty[s]` 上 arrive，这个 buffer 回到 producer 手里（[[cutlass:include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp@mma|collective mma]]）。stage 下标和 phase 由 [[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@PipelineState|PipelineState]] 维护：下标转一圈，phase 翻一次。
:::

## Warp specialization：不同 warp 承担不同角色

一个 warp group 负责 TMA（producer），其他 warp group 负责 MMA（consumer）。producer 几乎不需要寄存器，consumer 需要大量寄存器放累加器，所以 Hopper 上还会用 `setmaxnreg` 重新分配：CUTLASS cooperative kernel 里 producer 降到 40 个、consumer 升到 232 个（[[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp@LoadRegisterRequirement|LoadRegisterRequirement / MmaRegisterRequirement]]）。D3 让你在 64K 寄存器的预算下检查这些数，并比较两种调度：

| 调度 | 做法 | 好处 |
|---|---|---|
| Cooperative | 两个 consumer warp group 合作算同一个输出 tile | tile 更大，单个 tile 更快 |
| Ping-pong | 两个 consumer warp group 交替处理不同 tile | 一个做 epilogue 时另一个在做 mainloop，epilogue 被隐藏 |

## PDL：跨 kernel 的提前启动

PDL 不是 kernel 内的流水线，它解决的是两个有依赖关系的 kernel 之间的启动时机（D4，[[cutlass:media/docs/cpp/dependent_kernel_launch.md|dependent_kernel_launch.md]]）：

| 角色 | 做什么 | 在 CUTLASS 里 |
|---|---|---|
| producer kernel | 到达合适位置时发出“可以启动后继”的信号（`cudaTriggerProgrammaticLaunchCompletion` / `griddepcontrol.launch_dependents`） | [[cutlass:include/cutlass/arch/grid_dependency_control.h@launch_dependent_grids|launch_dependent_grids]] |
| consumer kernel | 提前开始做与 producer 输出无关的工作；读依赖数据之前等待（`cudaGridDependencySynchronize` / `griddepcontrol.wait`） | [[cutlass:include/cutlass/arch/grid_dependency_control.h@wait_on_dependent_grids|wait_on_dependent_grids]] |
| host | 用 extensible launch API 设置 programmatic stream serialization；CUDA Graph 用 programmatic dependency edge | [[cutlass:media/docs/cpp/dependent_kernel_launch.md|dependent_kernel_launch.md]] |

::: callout warn PDL 的正确性边界
trigger **不表示** producer 的写入已经可见；`wait` 等的是 producer **完成**并把内存刷出去。consumer 在 `wait` 之前只能访问明确与 producer 无关的数据。PDL 不是取消依赖，也不是让两个有数据竞争的 kernel 自动安全并行。
:::

::: flow 两个串联 GEMM 的 PDL 时间关系
cfg cw=200 rh=96 nw=168
n p1 0 0 "gemm1 mainloop" "读 X、W1" k=acc
n trg 1 0 "launch_dependents" "trigger" k=warn
n p2 2 0 "gemm1 epilogue" "写 H" k=acc
n c1 1 1 "gemm2 setup" "提前启动" k=teal
n c2 2 1 "prefetch W2" "与 H 无关" k=teal
n w 3 1 "griddepcontrol.wait" "等 gemm1 结束" k=todo
n c3 3 0 "gemm2 mainloop" "读 H、W2" k=pass
e p1 -> trg
e trg -> p2
e trg -> c1 "PDL"
e c1 -> c2
e c2 -> w
e p2 -> w "刷出" dash
e w -> c3
cap: 文档的例子：`H = X × W₁` 之后 `Y = H × W₂`。consumer 读的 W₂ 和 H 无关，所以它可以提前启动、先预取 W₂，到真正要读 H 时再 wait。CUTLASS 的 [[cutlass:examples/63_hopper_gemm_with_weight_prefetch|example 63（weight prefetch）]] 就是这个思路。
:::

真实的 cooperative kernel 里，producer warp 在第一次读 GMEM 之前调用 `wait_on_dependent_grids()`，consumer 在处理最后一个 tile 时调用 `launch_dependent_grids()`（[[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp@wait_on_dependent_grids|wait 的位置]] · [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp@launch_dependent_grids|trigger 的位置]]）。

PDL 的真正收益只在**系统层面**才看得见：单个 GEMM 测不出来，要在 CUDA Graph 或 stream 里有一串依赖的 kernel、每个 kernel 的 launch 与 prologue 开销相对它的运行时间不可忽略时（小 batch 推理、decode 阶段的一串小 GEMM），把下一个 kernel 的启动、setup 和独立 operand 的预取藏到上一个 kernel 的尾巴里。CUTLASS 的 example 63 README 也提醒：大部分收益要在端到端应用里看，`overlap_ratio` 和 `prefetch_ratio` 要按端到端时间 autotune。D4 的 `best_trigger` 就是这个 autotune 的最小模型。

## 四种机制和 Layout Algebra 的耦合程度

| 机制 | 作用范围 | 回答的问题 | 和 Layout Algebra 的关系 |
|---|---|---|---|
| Layout Algebra | 编译期 / 数据映射 | 坐标、线程、value、地址怎样对应 | 核心对象 |
| TMA + SMEM layout | kernel 内搬运 | 一个 tile 怎样描述并搬到指定排布 | 紧密耦合 |
| Async pipeline / barrier | kernel 内时序 | 哪个 stage 何时 ready、何时可复用 | 中等耦合 |
| Warp specialization | kernel 内分工 | 哪些 warp 生产、哪些消费 | 中等耦合 |
| PDL | kernel 之间 | 后继 kernel 何时启动、在哪里等待 | 弱耦合 |

**Layout 是空间关系，pipeline 是 kernel 内的时间关系，PDL 是 kernel 之间的时间 / 依赖关系。** 设计时先用 Layout Algebra 解决数据映射，再用 TMA、pipeline、barrier 解决 kernel 内的重叠，最后看 kernel 之间有没有可以提前做的独立工作。

::: callout analogy 餐厅的三种协调
Layout 是“哪道菜放在哪个盘子、由哪个服务员端”；pipeline 是厨房里“上一道还在炒，下一道已经在切菜，但切好的菜要等锅空了才能下”；warp specialization 是“专门有人切菜、专门有人炒”；PDL 是“下一桌客人还没吃完上一道，厨房已经开始准备他们点的、和上一道无关的凉菜，但热菜要等上一道的锅腾出来”。
:::

## 和练习题的对应关系

| 题目 | 复刻了什么 | 源码 |
|---|---|---|
| D1 | TMA 描述符、box、swizzle、越界填零、expect_tx 字节数 | [[cutlass:include/cute/atom/copy_traits_sm90_tma.hpp|copy_traits_sm90_tma.hpp]] |
| D2 | mbarrier phase、PipelineState、full / empty 协议 | [[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@PipelineTmaAsync|PipelineTmaAsync]] · [[cutlass:include/cutlass/arch/barrier.h|barrier.h]] |
| D3 | producer / consumer warp group、`setmaxnreg`、cooperative vs ping-pong | [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp|pingpong kernel]] |
| D4 | PDL 串联两个 GEMM、正确性检查 | [[cutlass:include/cutlass/arch/grid_dependency_control.h|grid_dependency_control.h]] |
