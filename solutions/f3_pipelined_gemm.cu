// F3 · 多级 cp.async 流水线 + ldmatrix + 寄存器双缓冲：sgemm_sm80 的完整结构 — reference solution.
//
// 在 F2（单级 SMEM、gemm 直接从 SMEM 读）的基础上加三件事：
//   1. SMEM 开 STAGES 级环形缓冲，主循环里“算第 k 块的同时，第 k+STAGES-1 块在路上”（cp.async + commit / wait_group）。
//   2. SMEM → 寄存器改用 ldmatrix.x4（SM75_U32x4_LDSM_N），一条指令给 warp 搬 4 个 8×8 小矩阵，布局正好是 mma 要的。
//   3. 寄存器双缓冲：算 k_block 的同时，把 k_block+1 的 fragment 从 SMEM 读进来。
// 结构逐行对照 CUTLASS examples/cute/tutorial/sgemm_sm80.cu 的 gemm_device（L54–L310），区别：
//   BF16 输入 + FP32 累加（SM80_16x8x16_F32BF16BF16F32_TN），C 是 FP32 行主序，8 个 warp（2×4）而不是 4 个。
//
// 配置（RTX 5090，见知识页“配置决策”第 2、5、6 步）：
//   CTA tile 128×128×64，STAGES = 3（3 × 32 KB = 96 KB ≤ 99 KB），256 线程 = 2×4 个 warp，
//   每个 warp 64×32 的 warp tile，每线程 64 个 FP32 累加器。
// 前提：M % 128 == 0、N % 128 == 0、K % 64 == 0（不整除的情况留给 F4）。
//
// 运行：python3 tools/cu_drill.py run solutions/f3_pipelined_gemm.cu 4096 4096 4096
#include <cute/tensor.hpp>

#include "common.cuh"
#include "tiled_gemm_config.hpp"

using namespace cute;

namespace f3 {

// 五个映射都来自 gpu/tiled_gemm_config.hpp（和 F2 相同）；这里只改 Stages = 3。
using Cfg = cfg::Config<128, 128, 64, /*Stages=*/3, /*WarpM=*/2, /*WarpN=*/4>;
using TA = Cfg::TA;
using TB = Cfg::TB;
using TC = Cfg::TC;
using BM = Int<Cfg::BM>;
using BN = Int<Cfg::BN>;
using BK = Int<Cfg::BK>;
constexpr int kStages = Cfg::Stages;
using SmemLayoutA = Cfg::SmemLayoutA;
using SmemLayoutB = Cfg::SmemLayoutB;
using G2SCopy = Cfg::G2SCopy;
using TiledMma = Cfg::TiledMma;
using S2RAtom = Cfg::S2RAtom;
constexpr int kThreads = Cfg::Threads;
constexpr int kSmemBytes = Cfg::SmemBytes;

// ------------------------------------------------------------------ kernel
__global__ __launch_bounds__(kThreads) void gemm_kernel(int M, int N, int K, TA const* A, TB const* B, TC* C) {
  static_assert(size(TiledMma{}) == kThreads && size(G2SCopy{}) == kThreads, "thread counts must match");

  // ① 全局张量和这个 CTA 的 tile（A1、B1）
  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));   // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(K, Int<1>{}));   // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));   // (M,N)
  auto cta_tiler = make_shape(BM{}, BN{}, BK{});
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});   // (BM,BK,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});   // (BN,BK,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});   // (BM,BN)

  // ② SMEM 环形缓冲（A3）
  extern __shared__ __align__(128) unsigned char smem_raw[];
  TA* smemA = reinterpret_cast<TA*>(smem_raw);
  TB* smemB = smemA + cosize_v<SmemLayoutA>;
  Tensor sA = make_tensor(make_smem_ptr(smemA), SmemLayoutA{});   // (BM,BK,STAGES)
  Tensor sB = make_tensor(make_smem_ptr(smemB), SmemLayoutB{});   // (BN,BK,STAGES)

  // ③ GMEM → SMEM 的线程分工（A2）
  G2SCopy g2s;
  auto thr_g2s = g2s.get_slice(threadIdx.x);
  Tensor tAgA = thr_g2s.partition_S(gA);   // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_g2s.partition_D(sA);   // (CPY,CPY_M,CPY_K,STAGES)
  Tensor tBgB = thr_g2s.partition_S(gB);
  Tensor tBsB = thr_g2s.partition_D(sB);

  // ④ MMA 分工和累加器（A4）
  TiledMma mma;
  auto thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);                  // (MMA,MMA_M,MMA_N)
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));  // (MMA,MMA_M,MMA_K) 寄存器
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));  // (MMA,MMA_N,MMA_K)
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);            // (MMA,MMA_M,MMA_N) FP32 累加器
  clear(tCrC);

  // ⑤ SMEM → 寄存器：ldmatrix，按 MMA 的 A/B 布局“重排”出来的 TiledCopy
  auto s2r_a = make_tiled_copy_A(S2RAtom{}, mma);
  auto thr_s2r_a = s2r_a.get_slice(threadIdx.x);
  Tensor tXsA = thr_s2r_a.partition_S(sA);   // (CPY,MMA_M,MMA_K,STAGES)
  Tensor tXrA = thr_s2r_a.retile_D(tCrA);    // (CPY,MMA_M,MMA_K)
  auto s2r_b = make_tiled_copy_B(S2RAtom{}, mma);
  auto thr_s2r_b = s2r_b.get_slice(threadIdx.x);
  Tensor tXsB = thr_s2r_b.partition_S(sB);
  Tensor tXrB = thr_s2r_b.retile_D(tCrB);

  // [TODO ⑥⑦] 预取 STAGES-1 级；等第 0 级到齐并读第一个 k16；主循环（kb==KB-1 时切换 read 级并 wait、kb==0 时发下一级 load 并推进 read/write、每步先读 kb+1 的 fragment 再算 kb）
  // ⑥ 预取：先把前 STAGES-1 个 k-tile 发出去，每个一组（commit group）
  int k_tile_count = size<3>(tAgA);
  int k_tile_next = 0;
  CUTE_UNROLL
  for (int s = 0; s < kStages - 1; ++s) {
    copy(g2s, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, s));
    copy(g2s, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, s));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) ++k_tile_next;
  }

  int smem_read = 0;
  int smem_write = kStages - 1;
  Tensor tXsA_p = tXsA(_, _, _, smem_read);
  Tensor tXsB_p = tXsB(_, _, _, smem_read);
  constexpr int kBlocks = decltype(size<2>(tCrA))::value;   // BK / 16 = 4 个 k16 步

  // 等第 0 级到齐，先把它的第一个 k16 读进寄存器
  cp_async_wait<kStages - 2>();
  __syncthreads();
  copy(s2r_a, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
  copy(s2r_b, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));

  // ⑦ 主循环：每个 k-tile 4 个 k16 步；第 0 步发下一级加载，最后一步切换到下一级 SMEM
  CUTE_NO_UNROLL
  while (k_tile_count > -(kStages - 1)) {
    CUTE_UNROLL
    for (int kb = 0; kb < kBlocks; ++kb) {
      if (kb == kBlocks - 1) {
        tXsA_p = tXsA(_, _, _, smem_read);
        tXsB_p = tXsB(_, _, _, smem_read);
        cp_async_wait<kStages - 2>();
        __syncthreads();
      }
      int kb_next = (kb + 1) % kBlocks;
      copy(s2r_a, tXsA_p(_, _, kb_next), tXrA(_, _, kb_next));   // 寄存器双缓冲：先读下一步
      copy(s2r_b, tXsB_p(_, _, kb_next), tXrB(_, _, kb_next));
      if (kb == 0) {
        copy(g2s, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_write));
        copy(g2s, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_write));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) ++k_tile_next;
        smem_write = smem_read;
        smem_read = (smem_read == kStages - 1) ? 0 : smem_read + 1;
      }
      gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrC);           // 再算这一步
    }
  }

  // [/TODO]

  // ⑧ epilogue：累加器直接写回（FP32；F4 会改成 BF16 + 谓词）
  copy(tCrC, tCgC);
}

// ------------------------------------------------------------------ host
inline void launch(int M, int N, int K, TA const* A, TB const* B, TC* C, cudaStream_t stream = 0) {
  dim3 grid(M / BM::value, N / BN::value);
  gemm_kernel<<<grid, kThreads, kSmemBytes, stream>>>(M, N, K, A, B, C);
}

inline void setup() {
  CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
}

}  // namespace f3

int main(int argc, char** argv) {
  auto a = drill::parse(argc, argv);
  if (a.M % 128 || a.N % 128 || a.K % 64) {
    std::printf("{\"ok\": false, \"error\": \"F3 needs M,N %% 128 == 0 and K %% 64 == 0\"}\n");
    return 1;
  }
  std::vector<__nv_bfloat16> hA(size_t(a.M) * a.K), hB(size_t(a.N) * a.K);
  drill::fill_bf16(hA, 1);
  drill::fill_bf16(hB, 2);
  auto* dA = drill::to_device(hA);
  auto* dB = drill::to_device(hB);
  float* dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dC, size_t(a.M) * a.N * sizeof(float)));

  f3::setup();
  auto go = [&] { f3::launch(a.M, a.N, a.K, reinterpret_cast<f3::TA const*>(dA), reinterpret_cast<f3::TB const*>(dB), dC); };
  go();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(size_t(a.M) * a.N);
  CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  double err = drill::max_rel_err(got, drill::reference(a.M, a.N, a.K, dA, dB));
  float ms = drill::time_ms(go, a.iters);
  char extra[128];
  std::snprintf(extra, sizeof extra, ", \"smem_bytes\": %d, \"threads\": %d, \"stages\": %d", f3::kSmemBytes,
                f3::kThreads, f3::kStages);
  drill::report("f3_pipelined_128x128x64_s3", a, err, ms, extra);
  return 0;
}
