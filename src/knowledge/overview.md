lede: 这套练习的主线：先只从硬件需求出发设计一个 Tiled GEMM，看我们怎样“自然地发明”各种 layout；再看这些 layout 怎样用代数组合、CuTe 在编译期怎样化简和检查它们；最后把 TMA、pipeline、warp specialization 和 PDL 加进来，得到一个结构完整的现代 GEMM。
# 总览：一条主线、三类抽象、五个 track

## 先建立正确的心智模型

写 GEMM 时，数学公式只告诉我们**算什么**；高性能 kernel 还必须决定每个数据元素在 GPU 的存储层级和执行资源中**怎样流动**。CuTe 的中心抽象是 layout：把一个逻辑坐标映射到一个整数。这个整数可以是内存偏移、线程编号、线程内寄存器编号，或者 tile 内的位置。CuTe 把这些映射统一成可以组合、拆分、推导的对象（[[cutlass:media/docs/cpp/cute/01_layout.md|01_layout.md]]）。

::: callout key 三类抽象，三个问题
| 抽象 | 回答的问题 | 例子 |
|---|---|---|
| Layout（空间关系） | 数据怎样分布：哪个线程拿哪个元素、元素放在哪个地址 | `local_tile`、Copy TV layout、SMEM swizzle、MMA TV layout |
| Copy / MMA Atom（操作） | 执行什么硬件操作 | `cp.async`、TMA、`ldmatrix`、`mma.sync`、`wgmma` |
| Pipeline / PDL（时间关系） | 操作之间什么时候能重叠、什么时候必须等待 | mbarrier 多级流水线、warp specialization、PDL |

这三类相关，但不是同一层。Layout Algebra 只管第一类；把 pipeline 或 PDL 当成 layout，会在设计时混淆“在哪里”和“什么时候”。
:::

第一次出现的缩写：

- **CTA**：Cooperative Thread Array，一个 thread block。**SMEM**：shared memory。**GMEM**：global memory（显存）。
- **MMA**：Matrix Multiply-Accumulate，Tensor Core 指令，SM80 上是 `mma.sync`，SM90 上是 warp group 级的 WGMMA。
- **TV layout**：(thread, value) → 坐标的 layout，回答“第 t 个线程的第 v 个值是哪个元素”。
- **TMA**：Tensor Memory Accelerator，Hopper 起的硬件单元，按描述符异步搬运整块 tensor。
- **mbarrier**：shared memory 里的异步屏障，记录“到达数”和“待到字节数”，用于 TMA 和流水线同步。
- **PDL**：Programmatic Dependent Launch，让后一个 kernel 在前一个结束前提前启动、在读依赖数据前再等待。

## 贯穿全部练习的例子

用一个小到能手算、但结构完整的配置：

| 参数 | 值 |
|---|---|
| 问题 | `A[64,64] × B[64,64]ᵀ → C[64,64]`，A、B 都是 K 连续（B 按 N×K 存储） |
| CTA tile | 32×32，`BK = 16`（每个 CTA 的 K 循环 4 轮），共 4 个 CTA |
| 线程 | 每个 CTA 128 个线程 = 4 个 warp |
| MMA | SM80 `mma.sync.m16n8k16`（F32 += F16 · F16），TiledMMA 用 2×2 个 atom |
| Copy | `ThrLayout = (32,4):(4,1)`，`ValLayout = (1,4)`：每个线程搬 4 个连续 half |
| SMEM | `Swizzle<2,2,2> ∘ (32,16):(16,1)` |

Track D、E 会把同一个设计“升级”成 Hopper 风格：TMA + mbarrier 多级流水线 + producer / consumer warp group + PDL。

::: flow 一个 GEMM 的数据流
cfg cw=185 rh=96 nw=135
n g 0 0 "Global A, B" "原始矩阵" k=dark
n s 1 0 "Shared Memory" "swizzled tile" k=acc
n r 2 0 "Registers" "每线程 fragment" k=vio
n m 3 0 "Warp MMA" "Tensor Core" k=teal
n acc 4 0 "Accumulator" "寄存器里的部分和" k=pass
n c 4 1 "Global C" "写回" k=dark
e g -> s "① ②" anim
e s -> r "③ ④"
e r -> m
e m -> acc "⑤"
e acc -> c "epilogue"
cap: 每一个箭头都是一个“映射问题”：谁搬、搬到哪、怎么摆、谁拿哪个值、结果在谁手里。Track A 的五道题依次回答它们；结构对照 [[cutlass:examples/cute/tutorial/sgemm_sm80.cu|sgemm_sm80.cu]]。
:::

## 五个 track

::: flow 练习路线
cfg cw=200 rh=90 nw=164
n a 0 0.5 "A 从零设计" "五个映射 → GEMM" k=acc
n b 1 0 "B Layout Algebra" "重新表达 · 向量宽度" k=vio
n c 1 1 "C 编译期" "化简 · 合法性检查" k=teal
n d 2 0.5 "D 时间维度" "TMA · 流水线 · PDL" k=warn
n e 3 0.5 "E 完整 GEMM" "拼装 · 选配置" k=pass
e a -> b
e a -> c
e b -> d
e c -> d
e d -> e
cap: 学习顺序：先把空间映射吃透，再理解 tile 结构和硬件映射，再学时间维度，最后是系统级执行（PDL、CUDA Graph、调度和 autotuning）。
:::

| Track | 题目 | 对应的知识页 |
|---|---|---|
| A | A1 CTA tile · A2 Copy TV · A3 SMEM swizzle · A4 MMA fragment · A5 拼成完整 GEMM | [从零设计](from_scratch.html) |
| B | B1 用代数重写 · B2 向量宽度 | [Layout Algebra 与编译](algebra_compile.html) |
| C | C1 静态化简 · C2 编译期检查 | [Layout Algebra 与编译](algebra_compile.html) |
| D | D1 TMA box · D2 mbarrier 流水线 · D3 warp specialization · D4 PDL 串联 | [时间维度](time_dimension.html) |
| E | E1 完整的 Hopper 风格 GEMM · E2 配置搜索 | [真实代码导读](real_code.html) |

## 上机对拍

CuTe 的 layout、TiledCopy、TiledMMA 都能在主机上编译运行：`oracle/<stem>.cpp` 直接 `#include <cute/tensor.hpp>`，用 `clang++` 编译，结果写进 `tests/golden/<stem>.json`。D2 的 oracle 直接用 CUTLASS 的 `PipelineState`。没有 GPU 也能跑的只有这些“记账”部分；TMA、mbarrier、warp specialization、PDL 的运行时行为用照着 CUTLASS 源码写的协议与时间线模拟器，讲解页里标明“未在 GPU 上运行”。

```bash
./check.sh a1                 # 对拍 golden（不需要编译器）
tools/fetch_headers.sh        # 下载 CUTLASS、CCCL、CUDA 公共头文件
./check.sh oracle a1          # 用真实 CuTe 重新生成 golden
playground/run.sh playground/a1_cta_tiling.cpp
```

继续阅读：[从零设计](from_scratch.html) · [Layout Algebra 与编译](algebra_compile.html) · [时间维度](time_dimension.html) · [真实代码导读](real_code.html) · [参考资料](references.html)。
