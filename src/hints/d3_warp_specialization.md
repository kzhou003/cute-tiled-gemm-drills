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

## 从小例子开始：先在草稿纸上手算

这道题有两本账：寄存器账（函数 `launch_regs_per_thread` 、`validate_setmaxnreg` 、`register_plan` 、`accumulator_regs` ）和时间账（`schedule_cooperative` 、`schedule_pingpong` 、`steady_per_tile` ）。两本账都先用能写满一张纸的小数字算一遍，再换成测试里的数字，最后写成一般公式。

### 寄存器账：从“每线程几个”到“整个 CTA 几个”

① 小：只看“每线程”这一列

先不乘 128。三个 WG 启动时每线程都是 168，重分配后是 40、232、232。

```
启动：168 + 168 + 168 = 504
之后：40 + 232 + 232 = 504   ✓ 一样多
producer 让出 168−40 = 128
两个 consumer 各要 232−168 = 64
```

② 测试的默认例子：384 线程，40 / 232

每个 WG 有 128 个线程，所以把 ① 的每个数乘 128。

```
launch_total = 168 × 384 = 64512
final_total  = 128 × 504 = 64512
freed = 128 × 128 = 16384
needed = 2 × 128 × 64 = 16384
```

③ 边界：混用 40 / 240，以及 3 个 consumer 每线程那一列一旦超过 504（或 4 × 128 = 512），就越界了。

```
40 + 240 + 240 = 520 > 504 → E_BUDGET
512 线程：launch = 128
24 + 3x ≤ 512 → x ≤ 162.67 → 160
```

④ 一般

把 ①–③ 写成公式。

```
launch = ⌊65536 / threads / 8⌋ × 8
目标值 % 8 == 0 且 24 ≤ 目标值 ≤ 256
load ≤ launch ≤ mma
Σ 每个 WG 的目标值 ≤ WG 数 × launch
```

**✎** 草稿纸 1：384 线程（1 个 producer + 2 个 consumer）的寄存器账

```text
1) 启动时每线程几个？  65536 / 384 = 170.67 → 取整 170 → 按 8 对齐：170 // 8 = 21，21 × 8 = 168
   启动时整个 CTA 分到：168 × 384 = 64512        （SM 上还剩 65536 − 64512 = 1024 个没分出去）
2) 检查目标值：40 % 8 = 0 ✓   24 ≤ 40 ≤ 256 ✓      232 % 8 = 0 ✓   24 ≤ 232 ≤ 256 ✓
   方向：producer 用 dec，40 ≤ 168 ✓；consumer 用 inc，232 ≥ 168 ✓
3) 按 WG 记账（每 WG 128 线程）
           WG0 producer   168 → 40     放出 (168 − 40) × 128 = 128 × 128 = 16384
           WG1 consumer   168 → 232    取走 (232 − 168) × 128 =  64 × 128 =  8192
           WG2 consumer   168 → 232    取走 (232 − 168) × 128 =  64 × 128 =  8192
   freed = 16384      needed = 8192 + 8192 = 16384      → 刚好放出多少就用掉多少
4) 总量：final_total = 128 × (1·40 + 2·232) = 128 × 504 = 64512 ≤ 64512 ✓   → ok =
True，errors = []
5) 换成 40 / 240：  128 × (40 + 240 + 240) = 128 × 520 = 66560 > 64512 ✗
   → errors = ['E_BUDGET: final 66560 registers > launch allocation 64512']
   换成 24 / 240：  128 × (24 + 240 + 240) = 128 × 504 = 64512 ✓      （CUTLASS 的“重负载”配置）
6) 累加器：tile 128×128，fp32，每元素 1 个寄存器，共 16384 个
   Cooperative：256 个线程分  → 16384 / 256 = 64 个/线程
   Ping-pong  ：128 个线程独占 → 16384 / 128 = 128 个/线程     （都在 232 以内；≥ 208 才换成 24 / 240）
```

### 时间账：2 个 tile，两种调度各画一遍

取 `M = 3` （一个 tile 的 MMA 时间）、`E = 2` （一个 WG 写回一整个 tile 的时间）、`n = 2` 个 tile。草稿纸上横轴是时间，每格 1 个单位，每行是一个 WG。

① 小：n = 2，M = 3，E = 2

每个时间格都能写出来。结果两种调度一样长。

```
Cooperative：(3 + 1) × 2 = 8
Ping-pong  ：3 + 2 + 1 × 3 = 8
→ 只有 2 个 tile 时 Ping-pong 没赢
```

② 测试的例子：M = 10，E = 4

tile 多了以后差距才出来。

```
n = 4：Coop 48，Ping-pong 44
n = 8：Coop 96，Ping-pong 84
利用率 0.833 对 0.952
```

③ 边界：E > 2M（M = 4，E = 20）

epilogue 太长，Ping-pong 的 epilogue 链成了瓶颈。

```
每 tile：Coop 4 + 10 = 14
        Ping-pong max(4, 20) = 20
n = 4：Coop 56，Ping-pong 84
```

④ 一般

（已对 n ≤ 11、M ≤ 14、E ≤ 39 全部核对过参考解）

```
Coop：makespan = n·(M + E/2)
PP  ：makespan = M + E + (n−1)·max(M, E)
稳态每 tile：M + E/2 对 max(M, E)
PP 不慢 ⇔ E ≤ 2M
```

**✎** 草稿纸 2：n = 2，M = 3，E = 2

```text
── Cooperative：两个 WG 一起算同一个 tile，一起写回（各写一半，E/2 = 1）──
   tile 0：mma [0, 3)       两个 WG 都在这一段         epi [3, 4)  两个 WG 各写一半
   tile 1：mma 从上一个 epi 结束开始 = 4 → [4, 7)      epi [7, 8)
   t        0  1  2  3  4  5  6  7        ← 第 t 格 = 时间 [t, t+1)
   WG1      M0 M0 M0 E0 M1 M1 M1 E1
   WG2      M0 M0 M0 E0 M1 M1 M1 E1
   TC       ## ## ## .. ## ## ## ..        ← 写回时 Tensor Core 空着
   makespan = 8，tensor_core_busy = 2 × 3 = 6，utilization = 6 / 8 = 0.75
   segments 共 8 条：(1,mma,0,0,3) (2,mma,0,0,3) (1,epi,0,3,4) (2,epi,0,3,4)
(1,mma,1,4,7) ...
── Ping-pong：tile j 归 WG 1 + j%2，每个 WG 独自写回整个 tile（E = 2）──
   tile 0 → WG1： mma_start = max(mma_end[−1], epi_end[−2]) = max(0, 0) = 0     → mma [0, 3)
                  epi_start = max(mma_end[0], epi_end[−1])  = max(3, 0) = 3     → epi [3, 5)
   tile 1 → WG2： mma_start = max(mma_end[0], epi_end[−1])  = max(3, 0) = 3     → mma [3, 6)
                  epi_start = max(mma_end[1], epi_end[0])   = max(6, 5) = 6     → epi [6, 8)
   t        0  1  2  3  4  5  6  7        ← 第 t 格 = 时间 [t, t+1)
   WG1      M0 M0 M0 E0 E0 .. .. ..
   WG2      .. .. .. M1 M1 M1 E1 E1
   TC       ## ## ## ## ## ## .. ..        ← 中间不停，但最后一个 epi（2 格）露在外面
   makespan = 8，utilization = 6 / 8 = 0.75
   segments 按 (start, wg) 排序：(1,mma,0,0,3) (1,epi,0,3,5) (2,mma,1,3,6)
(2,epi,1,6,8)
── 结论：一样长。Cooperative 每个 tile 露出 E/2 = 1，共 2；Ping-pong 只露出最后一个 E = 2。
   再加 tile：n = 3 → 12 对 11；n = 4 → 16 对 14。每多一个 tile，Coop 多 M + E/2 = 4，PP多 max(M, E) = 3。
```

下面这张图可以任意改小参数（n ≤ 6、M ≤ 8、E ≤ 12）。每个时间格里写的是 M + tile 号（MMA）或 E + tile 号（epilogue）。把鼠标停在一段上，面板会写出这一段的起点是由哪两个 `max` 项决定的（被依赖的两段会加粗框出）；移开鼠标，面板打印两个函数的完整返回值。先对照上面的草稿纸，再把 n 改成 3、4，或者把 E 拉到 7 以上（E > 2M）看看谁更快。

::: raw
<figure class="flow snap"><div class="flow-head">手算草稿 · 两种调度的逐格时间线</div><img src="../figs/d3/d3_new-01.png" alt="手算草稿 · 两种调度的逐格时间线" loading="lazy" style="width:100%;aspect-ratio:1461/1293"></figure>
:::

## 图解详解：寄存器预算与两种调度

::: callout info 关于这一节的图
这些图是交互版讲解页的静态快照，显示的是默认参数下的状态。文中提到的按钮、滑块和悬停提示在这里不能操作；图上的数字和下方的文字、表格、黑底面板说的是同一组值，可以直接对照你的函数输出。
:::

这一节有四张可交互的图，每张对应一组要实现的函数。整道题等于下面这段伪代码里的“算账”部分：

```
plan = register_plan(load_wgs=1, load_regs=40, mma_wgs=2, mma_regs=232)   # 图 1：kernel 开
acc  = accumulator_regs(128, 128, 256 if cooperative else 128)            # 图 2：一个 cons
s    = schedule_cooperative(n, M, E) or schedule_pingpong(n, M, E)        # 图 3：持久化 CTA
per  = steady_per_tile(kind, M, E)                                        # 图 4：tile 很多
```

### 图 1 · 寄存器预算条：`launch_regs_per_thread` / `validate_setmaxnreg` /

`register_plan`

先看启动寄存器 `launch_regs_per_thread` ：编译器把 65536 均分给每个线程，再向下对齐到 8 的倍数。

| `threads` | `WG` 数 | `65536 / threads` | `// 8 × 8` | `launch_total` | 没分出去 |
|---|---|---|---|---|---|
| `128` | `1` | `512` | `512*` | `65536` | `0` |
| `256` | `2` | `256` | `256` | `65536` | `0` |
| `384` | `3` | `170.67` | `168` | `64512` | `1024` |
| `512` | `4` | `128` | `128` | `65536` | `0` |
| `640` | `5` | `102.4` | `96` | `61440` | `4096` |

\* 只是本题公式的值。真实硬件每线程最多 255 个寄存器，128 线程的 kernel 分不到 512 个；这道题只用到 256 线程及以上的情况。

下面的图把整个 SM 的 65536 个寄存器画成一根横条。第一根是启动时每个 WG 拿到的（都是 launch × 128）；第二根是 setmaxnreg 之后。竖虚线是启动时分到的总数 launch\_total：第二根条超过它就是 `E_BUDGET` （红色斜线）。下方的柱子是“每线程”视角：虚线是 launch，柱高是目标值，箭头向下是 dec（放出），向上是 inc（取走）。拖滑块（步长 4，所以也能拖到不是 8 的倍数的值），看错误码怎样出现。

::: raw
<figure class="flow snap"><div class="flow-head">register_plan(load_wgs, load_regs, mma_wgs, mma_regs)</div><img src="../figs/d3/d3_new-02.png" alt="register_plan(load_wgs, load_regs, mma_wgs, mma_regs)" loading="lazy" style="width:100%;aspect-ratio:1461/1375"></figure>
:::

源码对照：`setmaxnreg` 只是两行内联 PTX，见 [[cutlass:include/cutlass/arch/reg_reconfig.h#L81-L95|warpgroup_reg_alloc / warpgroup_reg_dealloc]] ；两份 kernel 各在 producer 分支开头 dec（[[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp#L562|cooperative L562]] 、[[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp#L577|pingpong L577]] ），在 consumer 分支开头 inc（[[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp#L770-L772|cooperative L772]] 、[[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp#L805|pingpong L805]] ）。指令本身的规则（8 的倍数、[24, 256]、整个 warp group 一起执行）见 PTX ISA 的 [setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg) 一节。

### 图 2 · 累加器住在哪：`accumulator_regs`

consumer 需要 232 个寄存器，主要是为了放累加器。一个 tile\_m × tile\_n 的 fp32 tile 共有 tile\_m·tile\_n 个累加器，平均分给参与 MMA 的线程。Cooperative 的 TiledMMA 用 256 线程（两个 WG 沿 M 各分一半），Ping-pong 一个 WG（128 线程）独占整个 tile。右边的柱子是一个 consumer 线程的 232 个寄存器里，累加器占了多少。

::: raw
<figure class="flow snap"><div class="flow-head">accumulator_regs(tile_m, tile_n, threads)</div><img src="../figs/d3/d3_new-03.png" alt="accumulator_regs(tile_m, tile_n, threads)" loading="lazy" style="width:100%;aspect-ratio:1461/972"></figure>
:::

源码对照：kernel 里的同一个公式是 [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp#L131-L151|RegsPerThread 与 HeavyRegisterPressure]] （≥ 208 改用 24 / 240）；Cooperative 沿 M 切成两半来自 builder 的 [[cutlass:include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl#L252-L256|AtomLayoutMNK = 2×1×1]] ，kernel 还用 [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp#L373-L375|static_assert]] 要求 256 线程、tile\_m ≥ 128。

### 图 3 · 两种调度的甘特图：`schedule_cooperative` / `schedule_pingpong`

换成测试的数字（默认 M = 10、E = 4、n = 4）。上半是 Cooperative，下半是 Ping-pong；每块的第三行是 Tensor Core。点“▶ 播放”，竖线沿时间走，面板告诉你此刻每个 WG 在干什么、有序屏障在等谁。要看的三件事：

- Cooperative 的 Tensor Core 行每个 tile 之后都有一段 E/2 的空闲；Ping-pong 的 Tensor Core 行在 E ≤ M 时连成一片，只在最后露出一个 E。
- Ping-pong 里同一个 WG 的段从不重叠：WG1 的 tile 2 必须等它自己写完 tile 0（`epi_end[j−2]` ），因为累加器寄存器还占着。
- 切到 E = 16 或 E = 20：Ping-pong 的 MMA 开始“等 epilogue”，Tensor Core 行出现空洞，瓶颈从 MMA 链换成了 epilogue 链。

::: raw
<figure class="flow snap"><div class="flow-head">schedule_cooperative / schedule_pingpong</div><img src="../figs/d3/d3_new-04.png" alt="schedule_cooperative / schedule_pingpong" loading="lazy" style="width:100%;aspect-ratio:1461/1030"></figure>
:::

Ping-pong 里那两个 `max` 就是有序屏障的两次 `wait` ：MMA 前的 `wait` 等另一个 WG 把 stage 0（MMA 位）交过来，epilogue 前的 `wait` 等另一个 WG 把 stage 1（epilogue 位）交过来；而“等自己上一个 tile 写完”是同一个 WG 的代码顺序自然保证的。对照 [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp#L155-L158|StagesPerMathWarpGroup = 2]] 、[[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp#L497-L501|group_id = WG 号 − Consumer0]] ，以及 kernel 里的 [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp#L851-L878|MMA 段的 wait / arrive]] 和 [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp#L900-L937|epilogue 段的 wait / arrive]] 。Cooperative 没有这个屏障：两个 WG 在同一段代码里一起走（[[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp#L791-L845|mma 之后紧跟 epilogue.store]] ）。三种 kernel 设计的官方说明见 [[cutlass:media/docs/cpp/efficient_gemm.md#L231-L250|efficient_gemm.md 的 Hopper Warp Specialization 一节]]。

### 图 4 · 稳态每 tile：`steady_per_tile` ，以及 E = 2M 的交叉点

tile 很多时，makespan ≈ n × 每 tile 时间。横轴是 E，纵轴是每 tile 的时间：Cooperative 是一条斜率 1/2 的直线 `M +` `E/2` ；Ping-pong 先平（`M` ，MMA 链是瓶颈）再以斜率 1 上升（`E` ，epilogue 链是瓶颈）。两条线在 E = 2M 相交。圆点是直接跑模拟得到的 `(makespan(n+2) − makespan(n)) / 2` （n = 6），应当正好落在线上——这就是 hint 2 里建议你自己做的验证。

::: raw
<figure class="flow snap"><div class="flow-head">steady_per_tile(kind, M, E)</div><img src="../figs/d3/d3_new-05.png" alt="steady_per_tile(kind, M, E)" loading="lazy" style="width:100%;aspect-ratio:1461/905"></figure>
:::

### 速查

| 函数 | 公式 | 默认例子 |
|---|---|---|
| `launch_regs_per_thread (t)` | (65536 // t) // 8 × 8 | 384 → 168，512 → 128 |
| `validate_setmaxnreg(c)` | c % 8 ≠ 0 → E\_NREG\_MULT8；c 不在 [24, 256] 内 → E\_NREG\_RANGE（两条可以同时出现） | 44 → MULT8；16 → RANGE |
| `register_plan(...)` | threads = 128·(L+C)；launch\_total = launch·threads；final = 128·(L·l + C·c)；freed = 128·L·(launch − l)；needed = 128·C·(c −launch)；错误顺序 MULT8/RANGE(l) → MULT8/RANGE(c) → DEALLOC\_ABOVE → ALLOC\_BELOW → BUDGET | 64512 / 64512/ 16384 /16384，ok |
| `accumulator_regs(m, n, t)` | m·n // t | 128×128：Coop 64，PP 128 |
| `schedule_cooperative` | t 游标：mma [t, t+M] ×2 WG，epi [t+M, t+M+E/2] ×2 WG；makespan = n·(M + E/2) | (4, 10, 4) → 48 |
| `schedule_pingpong` | wg = 1 + j%2；s = max(mma\_end[j−1], epi\_end[j−2])；e = max(mma\_end[j], epi\_end[j−1])；makespan = M + E + (n−1)·max(M,E) | (4, 10, 4) → 44 |
| 共同收尾 | 按 (start, wg) 稳定排序；busy = n·M；utilization = busy / makespan（makespan = 0 时为 0） | (8, 10, 4)：0.833 / 0.952 |
| `steady_per_tile` | cooperative → M + E/2；pingpong → max(M, E)；其它 → ValueError | (10, 16)：18.0 /16 |

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

## 真实代码

寄存器常数，简化自 [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp#L139-L148|pingpong kernel]] （Cooperative 在 [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp#L146-L151|L146–151]] ，常数相同）：

```
static_assert(NumMMAThreads == 128, "Pingpong kernel must have TiledMMA operating using 12
static_assert(MaxThreadsPerBlock == 384, "Pingpong kernel must have 384 threads in total."
static constexpr int RegsPerThread = (TileM * TileN * sizeof(ElementAccumulator)) / (NumMM
static constexpr bool HeavyRegisterPressure = RegsPerThread >= 208;
static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
static constexpr uint32_t MmaRegisterRequirement  = !HeavyRegisterPressure ? 232 : 240;
```

`RegsPerThread` 就是本题的 `accumulator_regs` 。角色分派和两条有序链，简化自 [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp#L576-L937|pingpong operator()]] ：

```
if (warp_group_role == WarpGroupRole::Producer) {
  cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();   // setmaxnreg.dec 40
  ...                                                                // TMA 发射、调度器、ep
} else {                                                             // Consumer0 / Consum
  cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();      // setmaxnreg.inc 232
  while (work_tile_info.is_valid()) {
    math_wg_order_barrier.wait();   collective_mainloop.mma(...);     math_wg_order_barrie
    math_wg_order_barrier.wait();   collective_epilogue.store(...);   math_wg_order_barrie
    ...                             // 跳到下一个属于自己的 tile（隔一个）
  }
}
```

有序屏障的两个关键行，简化自 [[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp#L1281-L1372|OrderedSequenceBarrier]] ：`stage_({0, params.group_id == 0, 0})` （“Group 0 -starts with an opposite phase”），以及 `arrive()` 通知 `(group_id + 1) % Length` 后 `++stage_` 。`SequenceDepth =` `2` 的两个 stage 正好是 “MMA” 和 “epilogue”，所以两条链各自交替。

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
