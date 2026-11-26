// F2 · 第一个 Tensor Core GEMM：TiledCopy（cp.async 16 B）+ swizzle 过的 SMEM + TiledMMA（mma.sync）— your implementation (run ./check.sh f2).
//
// Track A 的五个映射在这里第一次变成真实的 GPU 代码：
//   ① CTA tile   local_tile(mA, (BM,BN,BK), (bm,bn,_))
//   ② Copy TV    G2SCopy.partition_S / partition_D：每线程一次 16 B 的 cp.async
//   ③ SMEM       Swizzle<3,3,3> ∘ (8,(8,8)):(8,(1,64))，一行 128 B
//   ④ MMA TV     TiledMma.partition_A / partition_B：从 SMEM 直接读（cute::gemm 内部用普通 LDS 搬进寄存器）
//   ⑤ 累加器     TiledMma.partition_C + make_fragment_C，K 全部走完才写回
// 只有一级 SMEM：每个 k-tile 都是“搬 → 等 → 算 → 等”，访存延迟完全露在外面。F3 再加流水线和 ldmatrix。
//
// 配置：128×128×64，Stages = 1（32 KB SMEM），256 线程 = 2×4 个 warp，每线程 64 个 FP32 累加器。
// 前提：M % 128 == 0、N % 128 == 0、K % 64 == 0。
//
// 运行：python3 tools/cu_drill.py run solutions/f2_tensorcore_gemm.cu 4096 4096 4096
#include <cute/tensor.hpp>

#include "common.cuh"
#include "tiled_gemm_config.hpp"

using namespace cute;

namespace f2 {

using Cfg = cfg::Config<128, 128, 64, /*Stages=*/1, /*WarpM=*/2, /*WarpN=*/4>;
using TA = Cfg::TA;
using TB = Cfg::TB;
using TC = Cfg::TC;

__global__ __launch_bounds__(Cfg::Threads) void gemm_kernel(int M, int N, int K, TA const* A, TB const* B, TC* C) {
  // ① 全局张量 → 这个 CTA 的 tile
  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));   // (M,K) K 连续
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(K, Int<1>{}));   // (N,K) K 连续
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));   // (M,N) N 连续
  // [TODO ①] local_tile：gA (BM,BK,k)、gB (BN,BK,k)、gC (BM,BN)
  DRILL_TODO("① local_tile：gA (BM,BK,k)、gB (BN,BK,k)、gC (BM,BN)");
  // [/TODO]

  // ③ SMEM（只有一级，32 KB，静态分配就够）
  __shared__ __align__(128) TA smemA[cosize_v<Cfg::SmemLayoutA>];
  __shared__ __align__(128) TB smemB[cosize_v<Cfg::SmemLayoutB>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), Cfg::SmemLayoutA{})(_, _, 0);   // (BM,BK)
  Tensor sB = make_tensor(make_smem_ptr(smemB), Cfg::SmemLayoutB{})(_, _, 0);   // (BN,BK)

  // ② GMEM → SMEM 的线程分工
  // [TODO ②] G2SCopy 的 get_slice / partition_S / partition_D：tAgA、tAsA、tBgB、tBsB
  DRILL_TODO("② G2SCopy 的 get_slice / partition_S / partition_D：tAgA、tAsA、tBgB、tBsB");
  // [/TODO]

  // ④ ⑤ MMA 分工：A、B 从 SMEM 分，C 在寄存器里
  // [TODO ③] TiledMma 的 partition_A / partition_B（直接对 SMEM）、partition_C、make_fragment_C，累加器清零
  DRILL_TODO("③ TiledMma 的 partition_A / partition_B（直接对 SMEM）、partition_C、make_fragment_C，累加器清零");
  // [/TODO]

  // [TODO ④] K 循环：cp.async → fence → wait<0> → __syncthreads → gemm(mma, …) → __syncthreads；最后写回
  DRILL_TODO("④ K 循环：cp.async → fence → wait<0> → __syncthreads → gemm(mma, …) → __syncthreads；最后写回");
  // [/TODO]
}

inline void launch(int M, int N, int K, TA const* A, TB const* B, TC* C, cudaStream_t stream = 0) {
  dim3 grid(M / Cfg::BM, N / Cfg::BN);
  gemm_kernel<<<grid, Cfg::Threads, 0, stream>>>(M, N, K, A, B, C);
}

}  // namespace f2

int main(int argc, char** argv) {
  auto a = drill::parse(argc, argv);
  if (a.M % 128 || a.N % 128 || a.K % 64) {
    std::printf("{\"ok\": false, \"error\": \"F2 needs M,N %% 128 == 0 and K %% 64 == 0\"}\n");
    return 1;
  }
  std::vector<__nv_bfloat16> hA(size_t(a.M) * a.K), hB(size_t(a.N) * a.K);
  drill::fill_bf16(hA, 1);
  drill::fill_bf16(hB, 2);
  auto* dA = drill::to_device(hA);
  auto* dB = drill::to_device(hB);
  float* dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dC, size_t(a.M) * a.N * sizeof(float)));

  auto go = [&] { f2::launch(a.M, a.N, a.K, reinterpret_cast<f2::TA const*>(dA), reinterpret_cast<f2::TB const*>(dB), dC); };
  go();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(size_t(a.M) * a.N);
  CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  double err = drill::max_rel_err(got, drill::reference(a.M, a.N, a.K, dA, dB));
  float ms = drill::time_ms(go, a.iters);
  drill::report("f2_tensorcore_128x128x64_s1", a, err, ms);
  return 0;
}
