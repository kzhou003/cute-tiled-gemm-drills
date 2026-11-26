// Shared host/device helpers for the Track F programs (F1–F5).
//
// Every program: parse "M N K [iters]" → fill BF16 inputs deterministically → run its kernel →
// compare with a naive reference kernel (one thread per output, FP32 accumulation) → time it →
// print exactly one JSON line per kernel:
//   {"ok": true, "kernel": "...", "M": .., "N": .., "K": .., "max_rel_err": .., "ms": .., "tflops": .., ...}
// tools/cu_drill.py and the pytest GPU tests parse these lines.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr)                                                                          \
  do {                                                                                            \
    cudaError_t err_ = (expr);                                                                    \
    if (err_ != cudaSuccess) {                                                                    \
      std::printf("{\"ok\": false, \"error\": \"%s at %s:%d\"}\n", cudaGetErrorString(err_), __FILE__, \
                  __LINE__);                                                                      \
      std::exit(1);                                                                               \
    }                                                                                             \
  } while (0)

// mine/*.cu marks unwritten parts with DRILL_TODO("…"): a static_assert that always fails, so the
// compiler lists every remaining TODO with its description.
#define DRILL_TODO(msg) static_assert(::drill::kTodo, "DRILL_TODO: " msg)

namespace drill {

inline constexpr bool kTodo = false;

struct Args {
  int M, N, K, iters;
};

// argv: M N K [iters]   (defaults 1024 1024 1024 20)
inline Args parse(int argc, char** argv) {
  Args a{1024, 1024, 1024, 20};
  if (argc > 1) a.M = std::atoi(argv[1]);
  if (argc > 2) a.N = std::atoi(argv[2]);
  if (argc > 3) a.K = std::atoi(argv[3]);
  if (argc > 4) a.iters = std::atoi(argv[4]);
  if (a.M <= 0 || a.N <= 0 || a.K <= 0 || a.iters <= 0) {
    std::printf("{\"ok\": false, \"error\": \"M, N, K, iters must be positive\"}\n");
    std::exit(1);
  }
  return a;
}

// Deterministic values in [-1, 1) on a 1/64 grid: exactly representable in BF16, so the only
// rounding in a GEMM is in the FP32 accumulation.
inline void fill_bf16(std::vector<__nv_bfloat16>& v, unsigned seed) {
  unsigned x = 2463534242u ^ (seed * 0x9E3779B9u);
  for (auto& e : v) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    e = __float2bfloat16(float(int(x % 128u) - 64) / 64.0f);
  }
}

inline std::vector<float> to_float(std::vector<__nv_bfloat16> const& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = __bfloat162float(v[i]);
  return out;
}

template <class T>
T* to_device(std::vector<T> const& h) {
  T* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

// C[m][n] = Σ_k A[m][k] · B[n][k]   (A: M×K, B: N×K, both K-contiguous; C: M×N row-major FP32)
__global__ void reference_kernel(int M, int N, int K, __nv_bfloat16 const* A, __nv_bfloat16 const* B, float* C) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += __bfloat162float(A[size_t(m) * K + k]) * __bfloat162float(B[size_t(n) * K + k]);
  C[size_t(m) * N + n] = acc;
}

inline std::vector<float> reference(int M, int N, int K, __nv_bfloat16 const* dA, __nv_bfloat16 const* dB) {
  float* dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dC, size_t(M) * N * sizeof(float)));
  dim3 block(32, 8), grid((N + 31) / 32, (M + 7) / 8);
  reference_kernel<<<grid, block>>>(M, N, K, dA, dB, dC);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> h(size_t(M) * N);
  CUDA_CHECK(cudaMemcpy(h.data(), dC, h.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(dC));
  return h;
}

// max|got − ref| / max(1, max|ref|); +inf if got has NaN / inf (same rule as the Python side).
inline double max_rel_err(std::vector<float> const& got, std::vector<float> const& ref) {
  if (got.size() != ref.size()) return INFINITY;
  double diff = 0, scale = 1;
  for (size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i])) return INFINITY;
    diff = std::fmax(diff, std::fabs(double(got[i]) - double(ref[i])));
    scale = std::fmax(scale, std::fabs(double(ref[i])));
  }
  return diff / scale;
}

// Average milliseconds per call over `iters` calls, after 2 warm-up calls.
template <class F>
float time_ms(F&& f, int iters) {
  for (int i = 0; i < 2; ++i) f();
  CUDA_CHECK(cudaGetLastError());
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventRecord(t0));
  for (int i = 0; i < iters; ++i) f();
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  CUDA_CHECK(cudaGetLastError());
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
  return ms / iters;
}

// One JSON line. `extra` is appended verbatim inside the object (start it with ", ").
inline void report(char const* kernel, Args const& a, double err, float ms, char const* extra = "",
                   double tol = 1e-3) {
  double tflops = 2.0 * a.M * a.N * double(a.K) / (ms * 1e-3) / 1e12;
  std::printf("{\"ok\": %s, \"kernel\": \"%s\", \"M\": %d, \"N\": %d, \"K\": %d, \"max_rel_err\": %.3e, "
              "\"tol\": %.0e, \"ms\": %.4f, \"tflops\": %.2f%s}\n",
              err <= tol ? "true" : "false", kernel, a.M, a.N, a.K, err, tol, ms, tflops, extra);
  std::fflush(stdout);
}

}  // namespace drill
