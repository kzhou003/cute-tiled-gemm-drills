// F1 · 第一个真实的 CuTe kernel：local_tile + local_partition + SMEM，用 CUDA core（FFMA）算 — your implementation (run ./check.sh f1).
//
// 目的是先熟悉“CuTe 张量”在 GPU 上怎么用，还不碰 Tensor Core：
//   make_tensor(make_gmem_ptr(p), shape, stride)  全局张量（只是指针 + layout，不搬数据）
//   local_tile(mA, tiler, coord, Step<...>)       这个 CTA 的那一块（A1）
//   local_partition(tile, thr_layout, tid)        按线程布局把一块分给各个线程（A2 的“谁搬哪个”）
//   copy(src, dst) / gemm(a, b, c)                逐元素搬运 / 逐元素乘加（UniversalFMA）
// 结构对照 CUTLASS examples/cute/tutorial/sgemm_1.cu（gemm_device L44–L246，gemm_tn L300–L350），区别：
//   BF16 输入（读进 SMEM 时转成 float）、C 是行主序 FP32。
//
// 配置：CTA tile 128×128×8，256 线程。
//   搬运：tA = tB = (32,8) 行主序（相邻 8 个线程读同一行的 8 个 K 元素）
//   计算：tC = (16,16) 列主序，每线程负责 8×8 = 64 个输出，按 16 的步长散开（raked）
// 前提：M % 128 == 0、N % 128 == 0、K % 8 == 0。
//
// 运行：python3 tools/cu_drill.py run solutions/f1_simt_cute_gemm.cu 2048 2048 2048
#include <cute/tensor.hpp>

#include "common.cuh"

using namespace cute;

namespace f1 {

using TA = cute::bfloat16_t;
using TB = cute::bfloat16_t;
using TC = float;

using BM = Int<128>;
using BN = Int<128>;
using BK = Int<8>;
using ThrA = Layout<Shape<_32, _8>, Stride<_8, _1>>;     // (m,k) → 线程号，k 连续
using ThrB = Layout<Shape<_32, _8>, Stride<_8, _1>>;     // (n,k) → 线程号
using ThrC = Layout<Shape<_16, _16>, Stride<_1, _16>>;   // (m,n) → 线程号，m 连续
using SmemA = Layout<Shape<BM, BK>, Stride<BK, _1>>;     // (m,k) → SMEM 下标，k 连续
using SmemB = Layout<Shape<BN, BK>, Stride<BK, _1>>;
constexpr int kThreads = 256;

__global__ __launch_bounds__(kThreads) void gemm_kernel(int M, int N, int K, TA const* A, TB const* B, TC* C) {
  // 全局张量
  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));   // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(K, Int<1>{}));   // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));   // (M,N)

  // 这个 CTA 的 tile
  // [TODO ①] 用 local_tile 切出这个 CTA 的 gA (BM,BK,k)、gB (BN,BK,k)、gC (BM,BN)
  DRILL_TODO("① 用 local_tile 切出这个 CTA 的 gA (BM,BK,k)、gB (BN,BK,k)、gC (BM,BN)");
  // [/TODO]

  // SMEM 存 float：BF16 在搬进来时转换一次，后面的 FFMA 就不用每次再转
  __shared__ float smemA[cosize_v<SmemA>];
  __shared__ float smemB[cosize_v<SmemB>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), SmemA{});   // (BM,BK)
  Tensor sB = make_tensor(make_smem_ptr(smemB), SmemB{});   // (BN,BK)

  // 搬运分工：线程 t 负责 tile 里 (t/8 + 32i, t%8) 这几个元素
  // [TODO ②] 用 local_partition 分出搬运分工 tAgA/tAsA/tBgB/tBsB、计算分工 tCsA/tCsB/tCgC，建寄存器累加器 tCrC 并清零
  DRILL_TODO("② 用 local_partition 分出搬运分工 tAgA/tAsA/tBgB/tBsB、计算分工 tCsA/tCsB/tCgC，建寄存器累加器 tCrC 并清零");
  // [/TODO]

  // [TODO ③] K 循环（copy → __syncthreads → gemm → __syncthreads），最后把累加器写回 gC
  DRILL_TODO("③ K 循环（copy → __syncthreads → gemm → __syncthreads），最后把累加器写回 gC");
  // [/TODO]
}

inline void launch(int M, int N, int K, TA const* A, TB const* B, TC* C, cudaStream_t stream = 0) {
  dim3 grid(M / BM::value, N / BN::value);
  gemm_kernel<<<grid, kThreads, 0, stream>>>(M, N, K, A, B, C);
}

}  // namespace f1

int main(int argc, char** argv) {
  auto a = drill::parse(argc, argv);
  if (a.M % 128 || a.N % 128 || a.K % 8) {
    std::printf("{\"ok\": false, \"error\": \"F1 needs M,N %% 128 == 0 and K %% 8 == 0\"}\n");
    return 1;
  }
  std::vector<__nv_bfloat16> hA(size_t(a.M) * a.K), hB(size_t(a.N) * a.K);
  drill::fill_bf16(hA, 1);
  drill::fill_bf16(hB, 2);
  auto* dA = drill::to_device(hA);
  auto* dB = drill::to_device(hB);
  float* dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dC, size_t(a.M) * a.N * sizeof(float)));

  auto go = [&] { f1::launch(a.M, a.N, a.K, reinterpret_cast<f1::TA const*>(dA), reinterpret_cast<f1::TB const*>(dB), dC); };
  go();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(size_t(a.M) * a.N);
  CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  double err = drill::max_rel_err(got, drill::reference(a.M, a.N, a.K, dA, dB));
  float ms = drill::time_ms(go, a.iters);
  drill::report("f1_simt_128x128x8", a, err, ms);
  return 0;
}
