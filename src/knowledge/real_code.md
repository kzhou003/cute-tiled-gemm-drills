lede: 把“设计结构伪代码”换成真实的 CUTLASS 代码：先读 SM80 的 sgemm_sm80.cu（cp.async + ldmatrix + mma.sync + 多级 SMEM 流水线），再读 SM90 的 warp-specialized cooperative kernel（TMA + mbarrier + WGMMA + setmaxnreg + PDL）。每一段都标出它对应五个映射中的哪一个、哪个练习。
# 真实代码导读：从 sgemm_sm80 到 SM90 warp-specialized GEMM

## 伪代码，对应到哪里

```text
gA_tile = local_tile(gA, tile_MK, cta_coord)                 ← ① CTA tile        (A1)
tiled_copy = make_tiled_copy(copy_atom, thr_layout, val)     ← ② Copy TV         (A2, B1)
sA = make_tensor(smem, swizzled_layout_A)                    ← ③ SMEM swizzle    (A3, B2)
tiled_mma = make_tiled_mma(mma_atom, atom_layout_MNK)        ← ④⑤ MMA TV         (A4)
for k_tile: issue_load / wait_until_ready / gemm / release   ← pipeline          (D2, D3)
store(gC_tile, accumulator)                                  ← epilogue          (A5, E1)
// 后面还有依赖 kernel：PDL                                    ← launch 层          (D4)
```

## SM80：sgemm_sm80.cu

这个例子把五个映射写得很清楚，而且所有 layout 都是静态的。下面是 host 侧的配置（摘录自 [[cutlass:examples/cute/tutorial/sgemm_sm80.cu@gemm_tn|gemm_tn]]）：

```cpp
// ③ SMEM：swizzle atom，再 tile_to_shape 铺满 (BM, BK, PIPE)
auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                Layout<Shape <_8,Shape <_8, _8>>,
                                       Stride<_8,Stride<_1,_64>>>{});
auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));      // bM=128, bK=64, bP=3

// ② Copy：cp.async 16 字节一次，线程 16×8（K-major），每线程 1×8 个 half
TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                  Layout<Shape<_16,_8>,Stride<_8,_1>>{},
                                  Layout<Shape< _1,_8>>{});

// ④ MMA：m16n8k16 atom 2×2 排列，permutation 成 32×32×16（方便 LDSM）
TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                               Layout<Shape<_2,_2>>{},
                               Tile<_32,_32,_16>{});
```

和本练习的例子比较：练习用 `BM=BN=32, BK=16`、`ValLayout=(1,4)`（8 字节）、`Swizzle<2,2,2>`，是为了能手算；真实配置用更大的 tile、16 字节的 `cp.async` 和 `Swizzle<3,3,3>`，道理相同（B2 会解释为什么 16 字节需要 M=3）。

kernel 里，五个映射依次出现（摘录自 [[cutlass:examples/cute/tutorial/sgemm_sm80.cu@gemm_device|gemm_device]]）：

```cpp
// ① CTA tile：Step 投影选出 A 用 (M,K)、B 用 (N,K)、C 用 (M,N)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});   // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});   // (BLK_N,BLK_K,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});   // (BLK_M,BLK_N)
// ② 每个线程的 copy 分片：源在 GMEM，目标在 SMEM（含 PIPE 维）
Tensor tAgA = thr_copy_a.partition_S(gA);                             // (CPY,CPY_M,CPY_K,k)
Tensor tAsA = thr_copy_a.partition_D(sA);                             // (CPY,CPY_M,CPY_K,PIPE)
// ④⑤ MMA 分片：A/B fragment 和累加器
Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                // (MMA,MMA_M,MMA_K)
Tensor tCrC = thr_mma.make_fragment_C(tCgC);                          // (MMA,MMA_M,MMA_N)
// SMEM → 寄存器：按 MMA 的 A 分布构造 ldmatrix 拷贝，再 retile 到 fragment
TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);            // s2r_atom_a = SM75_U32x4_LDSM_N
Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);
```

主循环是一个软件流水线：GMEM→SMEM 用 `cp.async` 提前 `K_PIPE_MAX−1` 个 stage，SMEM→寄存器提前一个 `k_block`，计算当前 `k_block`（简化自同一个函数）：

```cpp
while (k_tile_count > -(K_PIPE_MAX-1)) {
  for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
    if (k_block == K_BLOCK_MAX - 1) {
      tXsA_p = tXsA(_,_,_,smem_pipe_read);
      cp_async_wait<K_PIPE_MAX-2>(); __syncthreads();                 // 等下一个 stage 到齐
    }
    copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next)); // 预取下一个 k_block
    if (k_block == 0) {
      copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write)); // 发起后面 stage
      cp_async_fence();
      smem_pipe_write = smem_pipe_read; smem_pipe_read = (smem_pipe_read + 1) % K_PIPE_MAX;
    }
    gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);            // mma.sync，累加器一直在寄存器
  }
}
```

这里的同步是 `cp.async` 的 group 计数（`cp_async_wait<N>`）加 `__syncthreads()`，所有线程既是 producer 也是 consumer。SM90 把这两件事拆开。

## SM90：TMA + mbarrier + warp specialization + PDL

CUTLASS 的 cooperative kernel 把一个 CTA 分成一个 producer warp group 和两个 consumer warp group（[[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp|sm90_gemm_tma_warpspecialized_cooperative.hpp]]）。

::: flow 一个 SM90 cooperative CTA
cfg cw=240 rh=100 nw=160
n prod 0 0.5 "Producer WG" "TMA · 40 regs" k=acc
n ring 1 0.5 "SMEM 环形缓冲" "S 个 stage · SW128" k=teal
n c0 2 0 "Consumer0 WG" "WGMMA · 232 regs" k=pass
n c1 2 1 "Consumer1 WG" "WGMMA · 232 regs" k=pass
n epi 3 0.5 "Epilogue" "写回 C tile" k=dark
e prod -> ring "full[s]" anim
e ring -> c0
e ring -> c1
e c0 -> epi
e c1 -> epi
e c0:t -> prod:t "empty[s]" dash
cap: producer 等 `empty[s]`、发 TMA；consumer 等 `full[s]`、做 WGMMA、arrive `empty[s]`。寄存器由 `setmaxnreg` 在两种角色之间重新分配（[[cutlass:include/cutlass/arch/reg_reconfig.h|reg_reconfig.h]]）。
:::

角色与寄存器重分配（简化自 kernel 的 `operator()`）：

```cpp
static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
static constexpr uint32_t MmaRegisterRequirement  = !HeavyRegisterPressure ? 232 : 240;

if (warp_group_role == WarpGroupRole::Producer) {
  cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();   // producer 让出寄存器
  ...
  cutlass::arch::wait_on_dependent_grids();                          // PDL：第一次读 GMEM 之前
  collective_mainloop.load(...);                                     // TMA 循环
}
else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
  cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();      // consumer 拿到更多寄存器
  collective_mainloop.mma(...);                                      // WGMMA 循环
  if (scheduler.is_last_tile(work_tile_info))
    cutlass::arch::launch_dependent_grids();                         // PDL：允许下一个 kernel 启动
}
```

源码的注释把 PDL 的两个原则说得很直白：`wait_on_dependent_grids` 之前不能访问可能还没刷出的全局内存；`launch_dependent_grids` 的时机只影响性能、不影响正确性，太早会和当前 kernel 抢资源，太晚则延迟变长（[[cutlass:include/cutlass/arch/grid_dependency_control.h|grid_dependency_control.h]]）。

producer 的 TMA 循环（简化自 [[cutlass:include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp@load|collective load]]）：

```cpp
for ( ; k_tile_count > 0; --k_tile_count) {
  pipeline.producer_acquire(smem_pipe_write);          // 等 empty[s]，并 expect_tx
  BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
  int write_stage = smem_pipe_write.index();
  copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
  copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
  ++k_tile_iter;
  ++smem_pipe_write;                                   // PipelineState：下标 +1，绕回时翻 phase
}
```

consumer 的 WGMMA 循环（简化自 [[cutlass:include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp@mma|collective mma]]）：

```cpp
for ( ; k_tile_count > 0; --k_tile_count) {
  auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
  pipeline.consumer_wait(smem_pipe_read, barrier_token);    // 等 full[s] 的 phase
  warpgroup_arrive();
  cute::gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), accum);   // WGMMA
  warpgroup_commit_batch();
  warpgroup_wait<K_PIPE_MMAS>();                            // 最多留 K_PIPE_MMAS 批在飞
  pipeline.consumer_release(smem_pipe_release);             // arrive empty[s]
  ++smem_pipe_read; ++smem_pipe_release;
}
```

pipeline 本身（简化自 [[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@PipelineTmaAsync|PipelineTmaAsync]]）：

```cpp
void producer_acquire(uint32_t stage, uint32_t phase) {
  empty_barrier_ptr_[stage].wait(phase);
  if (params_.is_leader)
    full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);
}
void consumer_wait(uint32_t stage, uint32_t phase) {
  full_barrier_ptr_[stage].wait(phase);
}
```

## SM80 与 SM90 的对照

| 环节 | SM80（sgemm_sm80） | SM90（warp-specialized） | 练习 |
|---|---|---|---|
| GMEM → SMEM | 所有线程 `cp.async`，按 Copy TV 分工 | 一个线程发 TMA，描述符驱动 | A2、D1 |
| SMEM 排布 | `Swizzle<3,3,3>` atom + `tile_to_shape` | GMMA SMEM atom（[[cutlass:include/cute/atom/mma_traits_sm90_gmma.hpp@Layout_K_SW128_Atom|Layout_K_SW128_Atom]] 等），TMA 写入同样的 swizzle | A3、B2、D1 |
| SMEM → 寄存器 | `ldmatrix`（`make_tiled_copy_A` + `retile_D`） | WGMMA 直接读 SMEM 描述符，A 也可以来自寄存器 | A4 |
| 同步 | `cp_async_wait<N>` + `__syncthreads()` | mbarrier full / empty + phase | D2 |
| 角色 | 所有线程既搬又算 | producer WG + consumer WG，`setmaxnreg` 重分配寄存器 | D3 |
| kernel 之间 | 无 | `wait_on_dependent_grids` / `launch_dependent_grids` | D4 |

::: callout analogy 小饭馆和流水线厨房
SM80 的 kernel 像一家小饭馆：每个厨师自己去仓库拿菜、自己切、自己炒，大家靠“等所有人都拿完菜”（`__syncthreads`）对齐节奏。SM90 像流水线厨房：有专门的采购员（producer WG，开着货车 TMA 整箱运货），炒菜师傅（consumer WG）只看取菜窗口上的灯（mbarrier 的 phase）；灯亮就炒，炒完把空盘放回窗口（`empty[s]`）。采购员几乎不需要工作台（寄存器），所以把工作台让给炒菜师傅（`setmaxnreg`）。
:::

## Python：CuTe DSL 版本

CUTLASS 也提供 CuTe DSL 写的同类 kernel，例如 Ampere [[cutlass:examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py|tensorop_gemm.py]] 和 Hopper [[cutlass:examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py|dense_gemm.py]]。它们用同样的 `local_tile`、`make_tiled_copy`、`make_tiled_mma` 和 pipeline 抽象，只是用 Python 写，经 AST rewrite + tracing 编译（见 [Layout Algebra 与编译](algebra_compile.html)）。本机没有 GPU，这些例子没有运行。

## 和练习题的对应关系

| 题目 | 复刻了什么 | 源码 |
|---|---|---|
| E1 | 完整的 warp-specialized GEMM：TMA box → SW128 环形缓冲 → mbarrier → 两个 consumer WG → epilogue → PDL | [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp|cooperative kernel]] · [[cutlass:examples/cute/tutorial/hopper/wgmma_tma_sm90.cu|wgmma_tma_sm90.cu]] |
| E2 | tile / stage / 调度 / swizzle 的约束与代价模型 | [[cutlass:include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl|sm90_gmma_builder.inl]] |
