// F4 · 边界和 epilogue：M、N 不整除时的谓词加载，α·acc + β·C，FP32 → BF16 写回 — your implementation (run ./check.sh f4).
//
// 在 F3 的流水线 kernel 上改两处：
//   1. 加载：用 identity tensor 算出每个线程要搬的元素的“全局坐标”，行号 ≥ M（或 ≥ N）的那几条 cp.async 不发（copy_if）。
//      为什么越界的 SMEM 可以不管：A 的第 m 行只影响 C 的第 m 行；越界的行算出来的累加器根本不会被写回。
//      所以只需要“不读越界地址”，不需要把那部分 SMEM 清零。
//   2. epilogue：同样用 identity tensor 得到每个累加器对应的 (m, n)，在界内的才写：
//          D[m][n] = bf16( α · acc + β · C[m][n] )            （C 和 D 是同一块 BF16 内存，原地更新）
// K 仍然要求是 64 的倍数：K 方向的越界会把垃圾加进每一个输出，处理方法是把最后一个 k-tile 的越界部分清零
// （CUTLASS 的 residue 处理），留作进阶。
//
// 运行：python3 tools/cu_drill.py run solutions/f4_epilogue_residue.cu 1000 760 1024
#include <cute/tensor.hpp>

#include "common.cuh"
#include "tiled_gemm_config.hpp"

using namespace cute;

namespace f4 {

using Cfg = cfg::Config<128, 128, 64, /*Stages=*/3, /*WarpM=*/2, /*WarpN=*/4>;
using TA = Cfg::TA;
using TB = Cfg::TB;
using TD = cute::bfloat16_t;   // 输出 BF16
constexpr int kStages = Cfg::Stages;

__global__ __launch_bounds__(Cfg::Threads) void gemm_kernel(int M, int N, int K, TA const* A, TB const* B, TD* C,
                                                            float alpha, float beta) {
  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(K, Int<1>{}));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));
  // identity tensor：元素的“值”就是它自己的坐标，用同样的 local_tile / partition 切，就知道每个元素在哪
  Tensor cA = make_identity_tensor(shape(mA));   // (M,K) → (m,k)
  Tensor cB = make_identity_tensor(shape(mB));
  Tensor cC = make_identity_tensor(shape(mC));

  auto cta_tiler = make_shape(Int<Cfg::BM>{}, Int<Cfg::BN>{}, Int<Cfg::BK>{});
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});
  Tensor gcA = local_tile(cA, cta_tiler, cta_coord, Step<_1, X, _1>{});   // (BM,BK,k) 的坐标
  Tensor gcB = local_tile(cB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  Tensor gcC = local_tile(cC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  extern __shared__ __align__(128) unsigned char smem_raw[];
  TA* smemA = reinterpret_cast<TA*>(smem_raw);
  TB* smemB = smemA + cosize_v<Cfg::SmemLayoutA>;
  Tensor sA = make_tensor(make_smem_ptr(smemA), Cfg::SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smemB), Cfg::SmemLayoutB{});

  typename Cfg::G2SCopy g2s;
  auto thr_g2s = g2s.get_slice(threadIdx.x);
  Tensor tAgA = thr_g2s.partition_S(gA);    // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_g2s.partition_D(sA);
  Tensor tAcA = thr_g2s.partition_S(gcA);   // 同样的分法，得到每条 cp.async 的全局坐标
  Tensor tBgB = thr_g2s.partition_S(gB);
  Tensor tBsB = thr_g2s.partition_D(sB);
  Tensor tBcB = thr_g2s.partition_S(gcB);

  // [TODO ①] 用 tAcA / tBcB 的行坐标建加载谓词 tApA、tBpB（形状 (CPY_M, CPY_K)，K 方向 stride 0），并在下面的 load 里改用 copy_if
  DRILL_TODO("① 用 tAcA / tBcB 的行坐标建加载谓词 tApA、tBpB（形状 (CPY_M, CPY_K)，K 方向 stride 0），并在下面的 load 里改用 copy_if");
  // [/TODO]

  typename Cfg::TiledMma mma;
  auto thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCcC = thr_mma.partition_C(gcC);   // 每个累加器的 (m, n)
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
  Tensor tCrC = thr_mma.partition_fragment_C(gC);   // FP32 累加器（类型来自 MMA，不是来自 gC 的 BF16）
  clear(tCrC);

  auto s2r_a = make_tiled_copy_A(typename Cfg::S2RAtom{}, mma);
  auto thr_s2r_a = s2r_a.get_slice(threadIdx.x);
  Tensor tXsA = thr_s2r_a.partition_S(sA);
  Tensor tXrA = thr_s2r_a.retile_D(tCrA);
  auto s2r_b = make_tiled_copy_B(typename Cfg::S2RAtom{}, mma);
  auto thr_s2r_b = s2r_b.get_slice(threadIdx.x);
  Tensor tXsB = thr_s2r_b.partition_S(sB);
  Tensor tXrB = thr_s2r_b.retile_D(tCrB);

  int k_tile_count = size<3>(tAgA);
  int k_tile_next = 0;
  CUTE_UNROLL
  for (int s = 0; s < kStages - 1; ++s) {
    copy_if(g2s, tApA, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, s));
    copy_if(g2s, tBpB, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, s));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) ++k_tile_next;
  }

  int smem_read = 0;
  int smem_write = kStages - 1;
  Tensor tXsA_p = tXsA(_, _, _, smem_read);
  Tensor tXsB_p = tXsB(_, _, _, smem_read);
  constexpr int kBlocks = decltype(size<2>(tCrA))::value;

  cp_async_wait<kStages - 2>();
  __syncthreads();
  copy(s2r_a, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
  copy(s2r_b, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));

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
      copy(s2r_a, tXsA_p(_, _, kb_next), tXrA(_, _, kb_next));
      copy(s2r_b, tXsB_p(_, _, kb_next), tXrB(_, _, kb_next));
      if (kb == 0) {
        copy_if(g2s, tApA, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_write));
        copy_if(g2s, tBpB, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_write));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) ++k_tile_next;
        smem_write = smem_read;
        smem_read = (smem_read == kStages - 1) ? 0 : smem_read + 1;
      }
      gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrC);
    }
  }

  // [TODO ②] epilogue：用 tCcC 判断 (m,n) 是否在界内；D = bf16(α·acc + β·C)，β == 0 时不读 C
  DRILL_TODO("② epilogue：用 tCcC 判断 (m,n) 是否在界内；D = bf16(α·acc + β·C)，β == 0 时不读 C");
  // [/TODO]
}

inline void setup() {
  CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::SmemBytes));
}

inline void launch(int M, int N, int K, TA const* A, TB const* B, TD* C, float alpha, float beta,
                   cudaStream_t stream = 0) {
  dim3 grid((M + Cfg::BM - 1) / Cfg::BM, (N + Cfg::BN - 1) / Cfg::BN);   // 向上取整：最后一块不满
  gemm_kernel<<<grid, Cfg::Threads, Cfg::SmemBytes, stream>>>(M, N, K, A, B, C, alpha, beta);
}

}  // namespace f4

int main(int argc, char** argv) {
  auto a = drill::parse(argc, argv);
  float alpha = argc > 5 ? float(std::atof(argv[5])) : 1.5f;
  float beta = argc > 6 ? float(std::atof(argv[6])) : -0.5f;
  if (a.K % 64) {
    std::printf("{\"ok\": false, \"error\": \"F4 still needs K %% 64 == 0\"}\n");
    return 1;
  }
  std::vector<__nv_bfloat16> hA(size_t(a.M) * a.K), hB(size_t(a.N) * a.K), hC(size_t(a.M) * a.N);
  drill::fill_bf16(hA, 1);
  drill::fill_bf16(hB, 2);
  drill::fill_bf16(hC, 3);
  auto* dA = drill::to_device(hA);
  auto* dB = drill::to_device(hB);
  auto* dC = drill::to_device(hC);

  f4::setup();
  f4::launch(a.M, a.N, a.K, reinterpret_cast<f4::TA const*>(dA), reinterpret_cast<f4::TB const*>(dB),
             reinterpret_cast<f4::TD*>(dC), alpha, beta);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16> hD(hC.size());
  CUDA_CHECK(cudaMemcpy(hD.data(), dC, hD.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  // 参考：α·(A·Bᵀ) + β·C0，在 FP32 里算；输出是 BF16，所以容差放宽到 BF16 的精度（约 2⁻⁸）
  auto ref = drill::reference(a.M, a.N, a.K, dA, dB);
  auto c0 = drill::to_float(hC);
  for (size_t i = 0; i < ref.size(); ++i) ref[i] = alpha * ref[i] + beta * c0[i];
  double err = drill::max_rel_err(drill::to_float(hD), ref);

  // 计时用 β = 0（不读 C），每次都覆盖写，结果不用于校验
  auto go = [&] {
    f4::launch(a.M, a.N, a.K, reinterpret_cast<f4::TA const*>(dA), reinterpret_cast<f4::TB const*>(dB),
               reinterpret_cast<f4::TD*>(dC), alpha, 0.f);
  };
  float ms = drill::time_ms(go, a.iters);
  char extra[96];
  std::snprintf(extra, sizeof extra, ", \"alpha\": %g, \"beta\": %g", alpha, beta);
  drill::report("f4_residue_bf16_epilogue", a, err, ms, extra, /*tol=*/1e-2);
  return 0;
}
