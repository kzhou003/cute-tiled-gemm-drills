// F5 · 在 RTX 5090 上扫配置：tile、stage、warp 排列、split-K，并和 cuBLAS 对比 — your implementation (run ./check.sh f5).
//
// 主循环来自 gpu/pipelined_mainloop.cuh（就是 F3 的那一段，模板化了）。这一题要写的是：
//   1. split-K：grid 的第三维 z 负责 K 的第 z 段。做法是把 A、B 的指针往后挪 z·Kc 个元素、K 换成 Kc，
//      leading dimension 仍然是原来的 K——对 kernel 来说就是一个更“薄”的子问题。
//   2. split-K 的归约：S == 1 时直接写；S > 1 时每个 CTA 对 C 做 FP32 atomicAdd（C 先清零）。
//      这是知识页“配置决策”4.2 里的 atomic 方式：最快，但加法顺序不确定，结果可能差最后几位。
//   3. 扫描：对每个配置检查合法性（整除、SMEM ≤ 99 KB），跑正确性和计时，每个配置打印一行 JSON；最后跑 cuBLAS 作为基准。
//
// 运行：python3 tools/cu_drill.py run solutions/f5_config_sweep_splitk.cu 1024 1024 16384
//       （tools/cu_drill.py 会给 F5 自动加 -lcublas；没有 cuBLAS 头文件时跳过基准）
#include <cute/tensor.hpp>

#include "common.cuh"
#include "pipelined_mainloop.cuh"
#include "tiled_gemm_config.hpp"

#if __has_include(<cublas_v2.h>)
#include <cublas_v2.h>
#define DRILL_HAS_CUBLAS 1
#else
#define DRILL_HAS_CUBLAS 0
#endif

using namespace cute;

namespace f5 {

template <class Cfg>
__global__ __launch_bounds__(Cfg::Threads) void gemm_kernel(int M, int N, int K, int splits,
                                                            typename Cfg::TA const* A, typename Cfg::TB const* B,
                                                            float* C) {
  // [TODO ①] split-K：第 z 段 K 的 mA、mB（指针挪 z·Kc，形状用 Kc，leading dimension 仍是 K）
  DRILL_TODO("① split-K：第 z 段 K 的 mA、mB（指针挪 z·Kc，形状用 Kc，leading dimension 仍是 K）");
  // [/TODO]
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));

  auto cta_tiler = make_shape(Int<Cfg::BM>{}, Int<Cfg::BN>{}, Int<Cfg::BK>{});
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  extern __shared__ __align__(128) unsigned char smem_raw[];
  typename Cfg::TiledMma mma;
  auto thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrC = thr_mma.partition_fragment_C(gC);
  clear(tCrC);

  cfg::pipelined_mainloop<Cfg>(gA, gB, smem_raw, mma, tCrC);

  // [TODO ②] 归约：splits == 1 直接写；否则对每个累加器 atomicAdd 到 C
  DRILL_TODO("② 归约：splits == 1 直接写；否则对每个累加器 atomicAdd 到 C");
  // [/TODO]
}

struct Problem {
  int M, N, K, iters;
  __nv_bfloat16 const* A;
  __nv_bfloat16 const* B;
  float* C;
  std::vector<float> const* ref;
};

// ③ 跑一个配置：不合法就打印原因；合法就校验 + 计时
template <class Cfg>
void run(Problem const& p, int splits, char const* name) {
  bool legal = p.M % Cfg::BM == 0 && p.N % Cfg::BN == 0 && p.K % (splits * Cfg::BK) == 0 &&
               Cfg::SmemBytes <= 99 * 1024;
  if (!legal) {
    std::printf("{\"ok\": true, \"skipped\": true, \"kernel\": \"%s\", \"splits\": %d, \"smem_bytes\": %d}\n", name,
                splits, Cfg::SmemBytes);
    return;
  }
  CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel<Cfg>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::SmemBytes));
  dim3 grid(p.M / Cfg::BM, p.N / Cfg::BN, splits);
  auto go = [&] {
    if (splits > 1) CUDA_CHECK(cudaMemsetAsync(p.C, 0, size_t(p.M) * p.N * sizeof(float)));
    gemm_kernel<Cfg><<<grid, Cfg::Threads, Cfg::SmemBytes>>>(
        p.M, p.N, p.K, splits, reinterpret_cast<typename Cfg::TA const*>(p.A),
        reinterpret_cast<typename Cfg::TB const*>(p.B), p.C);
  };
  go();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(size_t(p.M) * p.N);
  CUDA_CHECK(cudaMemcpy(got.data(), p.C, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  double err = drill::max_rel_err(got, *p.ref);
  float ms = drill::time_ms(go, p.iters);
  char extra[192];
  std::snprintf(extra, sizeof extra,
                ", \"BM\": %d, \"BN\": %d, \"BK\": %d, \"stages\": %d, \"warps\": \"%dx%d\", \"splits\": %d, "
                "\"ctas\": %d, \"smem_bytes\": %d",
                Cfg::BM, Cfg::BN, Cfg::BK, Cfg::Stages, Cfg::WarpM, Cfg::WarpN, splits,
                int(grid.x * grid.y * grid.z), Cfg::SmemBytes);
  drill::report(name, drill::Args{p.M, p.N, p.K, p.iters}, err, ms, extra);
}

#if DRILL_HAS_CUBLAS
// C(M×N 行主序) = A(M×K 行主序) · B(N×K 行主序)ᵀ。换成 cuBLAS 的列主序：Cᵀ(N×M) = B · Aᵀ，
// 也就是 gemm(op(B)=T, op(A)=N, m=N, n=M, k=K)，三个 leading dimension 分别是 K、K、N。
// [TODO ③] cuBLAS 基准：把行主序的 C = A·Bᵀ 换成列主序的 Cᵀ = B·Aᵀ，写出 cublasGemmEx 的参数（m、n、k、op、ld）
DRILL_TODO("③ cuBLAS 基准：把行主序的 C = A·Bᵀ 换成列主序的 Cᵀ = B·Aᵀ，写出 cublasGemmEx 的参数（m、n、k、op、ld）");
// [/TODO]
#endif

}  // namespace f5

using C128x128s3 = cfg::Config<128, 128, 64, 3, 2, 4>;
using C128x128s2 = cfg::Config<128, 128, 64, 2, 2, 4>;
using C128x64s4 = cfg::Config<128, 64, 64, 4, 2, 2>;
using C64x64s4 = cfg::Config<64, 64, 64, 4, 2, 2>;
using C128x256s2 = cfg::Config<128, 256, 64, 2, 2, 4>;

int main(int argc, char** argv) {
  auto a = drill::parse(argc, argv);
  std::vector<__nv_bfloat16> hA(size_t(a.M) * a.K), hB(size_t(a.N) * a.K);
  drill::fill_bf16(hA, 1);
  drill::fill_bf16(hB, 2);
  auto* dA = drill::to_device(hA);
  auto* dB = drill::to_device(hB);
  float* dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dC, size_t(a.M) * a.N * sizeof(float)));
  auto ref = drill::reference(a.M, a.N, a.K, dA, dB);
  f5::Problem p{a.M, a.N, a.K, a.iters, dA, dB, dC, &ref};

  f5::run<C128x128s3>(p, 1, "128x128x64_s3_w2x4");
  f5::run<C128x128s2>(p, 1, "128x128x64_s2_w2x4");
  f5::run<C128x64s4>(p, 1, "128x64x64_s4_w2x2");
  f5::run<C64x64s4>(p, 1, "64x64x64_s4_w2x2");
  f5::run<C128x256s2>(p, 1, "128x256x64_s2_w2x4");
  for (int s : {2, 4, 5, 8}) f5::run<C128x128s3>(p, s, "128x128x64_s3_w2x4_splitk");
  for (int s : {2, 4}) f5::run<C64x64s4>(p, s, "64x64x64_s4_w2x2_splitk");
#if DRILL_HAS_CUBLAS
  f5::run_cublas(p);
#endif
  return 0;
}
