// The five mappings of a Tiled GEMM as CuTe types, shared by F2–F5 and by their oracles.
//
//   Config<BM, BN, BK, Stages, WarpM, WarpN>
//     TA, TB = bfloat16_t, TC = float          C = A·Bᵀ, A: M×K and B: N×K both K-contiguous
//     SmemLayoutA / B   ③  tile_to_shape(Swizzle<3,3,3> ∘ (8,(8,8)):(8,(1,64)), (BM|BN, BK, Stages))
//     G2SCopy           ②  cp.async 16 B; threads (Threads/8, 8):(8,1), values (1,8)
//     TiledMma          ④  SM80_16x8x16_F32BF16BF16F32_TN, warps (WarpM, WarpN), Tile<16·WarpM, 16·WarpN, 16>
//     S2RAtom           ⑤  ldmatrix.x4 (SM75_U32x4_LDSM_N), used with make_tiled_copy_A / _B
//     Threads, SmemBytes
//
// Same structure as gemm_tn() in CUTLASS examples/cute/tutorial/sgemm_sm80.cu, with BF16 inputs,
// FP32 accumulation and a configurable warp grid. Only the BK = 64 swizzle atom is provided
// (one SMEM row = 64 BF16 = 128 B, which is what Swizzle<3,3,3> is built for).
// This header compiles on the host too (the oracles include it to print the exact coordinates).
#pragma once

#include <cute/tensor.hpp>

namespace cfg {

using namespace cute;

template <int BM_, int BN_, int BK_, int Stages_, int WarpM_, int WarpN_>
struct Config {
  static_assert(BK_ == 64, "the swizzle atom below is the 128 B (64 x BF16) one");
  static_assert(BM_ % (16 * WarpM_) == 0 && BN_ % (16 * WarpN_) == 0, "warp grid must tile the CTA tile");

  using TA = cute::bfloat16_t;
  using TB = cute::bfloat16_t;
  using TC = float;

  static constexpr int BM = BM_, BN = BN_, BK = BK_, Stages = Stages_, WarpM = WarpM_, WarpN = WarpN_;
  static constexpr int Threads = WarpM * WarpN * 32;
  static constexpr int SmemBytes = Stages * (BM + BN) * BK * int(sizeof(TA));

  // ③ SMEM: 8 rows x 64 columns per atom, 16 B chunks XOR-ed with the row (A3, D1)
  using SmemAtom = decltype(composition(Swizzle<3, 3, 3>{},
                                        Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{}));
  using SmemLayoutA = decltype(tile_to_shape(SmemAtom{}, make_shape(Int<BM>{}, Int<BK>{}, Int<Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(SmemAtom{}, make_shape(Int<BN>{}, Int<BK>{}, Int<Stages>{})));

  // ② GMEM -> SMEM: each thread one 16 B cp.async per pass; 8 threads cover one 64-wide row (A2)
  using G2SCopy = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, TA>{},
                                           Layout<Shape<Int<Threads / 8>, _8>, Stride<_8, _1>>{},
                                           Layout<Shape<_1, _8>>{}));

  // ④ MMA: WarpM x WarpN warps of m16n8k16; the N tile is 16 per warp so ldmatrix.x4 can feed B (A4)
  using TiledMma = decltype(make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                           Layout<Shape<Int<WarpM>, Int<WarpN>, _1>>{},
                                           Tile<Int<16 * WarpM>, Int<16 * WarpN>, _16>{}));

  // ⑤ SMEM -> registers
  using S2RAtom = Copy_Atom<SM75_U32x4_LDSM_N, TA>;
};

}  // namespace cfg
