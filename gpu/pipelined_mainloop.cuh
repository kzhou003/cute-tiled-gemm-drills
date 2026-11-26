// The F3 main loop (multi-stage cp.async pipeline + ldmatrix + register double buffering) as a
// template, so F5 can instantiate it for several configurations. Line-for-line the same as the
// [TODO ⑥⑦] part of solutions/f3_pipelined_gemm.cu; see that file and its hint page for why each
// wait / sync / read / write sits where it does.
//
//   pipelined_mainloop<Cfg>(gA, gB, smem, mma, acc)
//     gA (BM,BK,k) and gB (BN,BK,k): this CTA's tiles from local_tile; k-tile count = size<2>(gA)
//     smem: dynamic shared memory of at least Cfg::SmemBytes bytes, 128 B aligned
//     acc:  thr_mma.partition_fragment_C(gC), already cleared; accumulated in place
// Requires full tiles (M % BM == N % BN == K % BK == 0); F4 shows the predicated version.
#pragma once

#include <cute/tensor.hpp>

#include "tiled_gemm_config.hpp"

namespace cfg {

template <class Cfg, class TensorGA, class TensorGB, class TiledMma, class TensorAcc>
__device__ void pipelined_mainloop(TensorGA const& gA, TensorGB const& gB, unsigned char* smem_raw, TiledMma& mma,
                                   TensorAcc& tCrC) {
  using namespace cute;
  constexpr int kStages = Cfg::Stages;

  auto* smemA = reinterpret_cast<typename Cfg::TA*>(smem_raw);
  auto* smemB = smemA + cosize_v<typename Cfg::SmemLayoutA>;
  Tensor sA = make_tensor(make_smem_ptr(smemA), typename Cfg::SmemLayoutA{});   // (BM,BK,STAGES)
  Tensor sB = make_tensor(make_smem_ptr(smemB), typename Cfg::SmemLayoutB{});   // (BN,BK,STAGES)

  typename Cfg::G2SCopy g2s;
  auto thr_g2s = g2s.get_slice(threadIdx.x);
  Tensor tAgA = thr_g2s.partition_S(gA);   // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_g2s.partition_D(sA);   // (CPY,CPY_M,CPY_K,STAGES)
  Tensor tBgB = thr_g2s.partition_S(gB);
  Tensor tBsB = thr_g2s.partition_D(sB);

  auto thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));   // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));   // (MMA,MMA_N,MMA_K)

  auto s2r_a = make_tiled_copy_A(typename Cfg::S2RAtom{}, mma);
  auto thr_s2r_a = s2r_a.get_slice(threadIdx.x);
  Tensor tXsA = thr_s2r_a.partition_S(sA);
  Tensor tXrA = thr_s2r_a.retile_D(tCrA);
  auto s2r_b = make_tiled_copy_B(typename Cfg::S2RAtom{}, mma);
  auto thr_s2r_b = s2r_b.get_slice(threadIdx.x);
  Tensor tXsB = thr_s2r_b.partition_S(sB);
  Tensor tXrB = thr_s2r_b.retile_D(tCrB);

  // prefetch STAGES-1 k-tiles, one commit group each
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
        copy(g2s, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_write));
        copy(g2s, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_write));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) ++k_tile_next;
        smem_write = smem_read;
        smem_read = (smem_read == kStages - 1) ? 0 : smem_read + 1;
      }
      gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrC);
    }
  }
}

}  // namespace cfg
