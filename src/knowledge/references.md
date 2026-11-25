lede: 这套练习依据的学习笔记、CUTLASS / CuTe 官方文档与源码、CUDA 编程指南中的相关章节，以及相关的练习仓库。
# 参考资料与边界说明

## 本练习依据的文档

《从 Tiled GEMM 重新理解 CuTe：设计动机、Layout Algebra、编译器与执行优化》（学习笔记，2026-11）。本练习的主线、例子配置（64×64、32×32 CTA tile、`BK = 16`、128 线程、SM80 m16n8k16）和分层结论都来自它。文中的配置是教学示例，不代表任何架构上的最优配置。

## CuTe / CUTLASS 文档（固定在 CUTLASS af95104）

| 文档 | 内容 |
|---|---|
| [[cutlass:media/docs/cpp/cute/01_layout.md|01_layout.md]]、[[cutlass:media/docs/cpp/cute/02_layout_algebra.md|02_layout_algebra.md]] | Layout 与 Layout Algebra |
| [[cutlass:media/docs/cpp/cute/0t_mma_atom.md|0t_mma_atom.md]] | MMA atom 与 TV layout |
| [[cutlass:media/docs/cpp/cute/0x_gemm_tutorial.md|0x_gemm_tutorial.md]] | CuTe GEMM 教程 |
| [[cutlass:media/docs/cpp/cute/0z_tma_tensors.md|0z_tma_tensors.md]] | TMA tensor |
| [[cutlass:media/docs/cpp/pipeline.md|pipeline.md]] | CUTLASS 异步 pipeline |
| [[cutlass:media/docs/cpp/efficient_gemm.md|efficient_gemm.md]] | 高效 GEMM 的结构（warp specialization、ping-pong 等） |
| [[cutlass:media/docs/cpp/dependent_kernel_launch.md|dependent_kernel_launch.md]] | CUTLASS 中的 PDL |
| [[cutlass:media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.rst|dsl_code_generation.rst]]、[[cutlass:media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.rst|dsl_jit_caching.rst]] | CuTe DSL 的编译与 JIT 缓存 |

## CUTLASS 源码

| 主题 | 源码 |
|---|---|
| SM80 CuTe GEMM | [[cutlass:examples/cute/tutorial/sgemm_sm80.cu|sgemm_sm80.cu]] |
| SM90 TMA + WGMMA 教程 | [[cutlass:examples/cute/tutorial/hopper/wgmma_tma_sm90.cu|wgmma_tma_sm90.cu]] |
| Layout Algebra | [[cutlass:include/cute/layout.hpp|layout.hpp]] · [[cutlass:include/cute/swizzle.hpp|swizzle.hpp]] · [[cutlass:include/cute/swizzle_layout.hpp|swizzle_layout.hpp]] |
| Copy / MMA atom | [[cutlass:include/cute/atom/copy_atom.hpp|copy_atom.hpp]] · [[cutlass:include/cute/atom/mma_traits_sm80.hpp|mma_traits_sm80.hpp]] · [[cutlass:include/cute/atom/copy_traits_sm90_tma.hpp|copy_traits_sm90_tma.hpp]] · [[cutlass:include/cute/atom/mma_traits_sm90_gmma.hpp|mma_traits_sm90_gmma.hpp]] |
| 自动向量化 copy | [[cutlass:include/cute/algorithm/copy.hpp@AutoVectorizingCopyWithAssumedAlignment|AutoVectorizingCopyWithAssumedAlignment]] · [[cutlass:include/cute/arch/copy_sm80.hpp|copy_sm80.hpp]] |
| Pipeline 与 barrier | [[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp|sm90_pipeline.hpp]] · [[cutlass:include/cutlass/arch/barrier.h|barrier.h]] |
| Warp-specialized kernel | [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp|cooperative]] · [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp|pingpong]] · [[cutlass:include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp|collective mainloop]] |
| 寄存器重分配 | [[cutlass:include/cutlass/arch/reg_reconfig.h|reg_reconfig.h]] |
| PDL | [[cutlass:include/cutlass/arch/grid_dependency_control.h|grid_dependency_control.h]] · [[cutlass:examples/63_hopper_gemm_with_weight_prefetch|example 63（weight prefetch）]] |
| 配置选择 | [[cutlass:include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl|sm90_gmma_builder.inl]] |

## CUDA 编程指南与 PTX ISA

- [Programmatic Dependent Launch and Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization)：`cudaTriggerProgrammaticLaunchCompletion`、`cudaGridDependencySynchronize`、launch 属性。
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)：`griddepcontrol`、`mbarrier`、`cp.async.bulk.tensor`（TMA）、`setmaxnreg`、`mma.sync` m16n8k16 的 fragment 布局。

## 相关练习

- [cuda-kernels-dev-drills](https://github.com/kzhou003/cuda-kernels-dev-drills)：在 RTX 5090 上真机写 GEMM、FlashAttention 和各代 GPU 的优化（包括 PDL 的真实测量）。
- [gpu-kernel-drills](https://github.com/kzhou003/gpu-kernel-drills)：GPU kernel 的基础练习。
- [ai-compiler-drills](https://github.com/kzhou003/ai-compiler-drills)：AI 编译器的练习，包括 launch 时间线和融合。

## 边界说明

- CUTLASS / CuTe API 会随版本和 GPU 架构变化；具体的 atom、TMA 操作、MMA 指令、pipeline API 和 launch 参数，以目标版本的官方文档和示例为准。
- **本仓库的测试不需要 GPU。** Layout 相关的答案来自真实 CuTe 的主机端程序；D2 的 `PipelineState` 来自真实 CUTLASS 头文件；TMA、mbarrier、warp specialization 和 PDL 的运行时行为是照着 CUTLASS 源码写的协议与时间线模拟器，数字是模型数字，不是实测。
- 这些机制真正跑起来需要 Hopper（SM90）或更新的 GPU。讲解页里“在 GPU 上可以这样验证”的说法没有在本机核对。
