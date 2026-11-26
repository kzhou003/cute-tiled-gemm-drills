// The end-to-end example (knowledge page "端到端例子"): C[1024x256] = A[1024x1024] · B[256x1024]ᵀ with
// the chosen plan ④ — CTA tile 64x32x64, 4 warps, 4 stages, BF16 in / FP32 accumulate. Prints the
// five mappings with real CuTe on the host: the same calls as sgemm_sm80.cu's TN configuration,
// with a smaller tile and BF16.
#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;

template <class T>
void starts(char const* name, T const& part) {   // first coordinate of every 16 B chunk
  printf("%s", name);
  for (int i = 0; i < size<1>(part); ++i) printf(" (%d,%d)", int(get<0>(part(0, i, 0))), int(get<1>(part(0, i, 0))));
}

int main() {
  int M = 1024, N = 256, K = 1024;
  auto bM = Int<64>{}; auto bN = Int<32>{}; auto bK = Int<64>{}; auto bP = Int<4>{};

  // ① CTA tile
  auto mA = make_identity_tensor(make_shape(M, K));
  auto mB = make_identity_tensor(make_shape(N, K));
  auto gA = local_tile(mA, make_shape(bM, bK), make_coord(3, _));
  auto gB = local_tile(mB, make_shape(bN, bK), make_coord(5, _));
  printf("① grid = (%d, %d) = %d CTAs\n", ceil_div(M, 64), ceil_div(N, 32), ceil_div(M, 64) * ceil_div(N, 32));
  printf("   gA shape "); print(shape(gA)); printf("  gA(0,0,2) = (%d,%d)\n", int(get<0>(gA(0, 0, 2))), int(get<1>(gA(0, 0, 2))));
  printf("   gB shape "); print(shape(gB)); printf("\n");

  // ② Copy TV: ThrLayout (16,8):(8,1), ValLayout (1,8), 16 B cp.async
  auto copy = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bfloat16_t>{},
                              Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
  auto tA = make_identity_tensor(make_shape(bM, bK));
  auto tB = make_identity_tensor(make_shape(bN, bK));
  printf("② copy starts (A tile 64x64 | B tile 32x64)\n");
  for (int t : {0, 1, 8, 9, 127}) {
    printf("   t=%3d ", t);
    starts("A:", copy.get_slice(t).partition_S(tA));
    starts("  B:", copy.get_slice(t).partition_S(tB));
    printf("\n");
  }

  // ③ SMEM: Swizzle<3,3,3> ∘ (8,(8,8)):(8,(1,64)) tiled to (64,64,4) and (32,64,4)
  auto atom = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{});
  auto sA = tile_to_shape(atom, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(atom, make_shape(bN, bK, bP));
  printf("③ sA cosize %d, sB cosize %d elements (%d KB total)\n", int(cosize(sA)), int(cosize(sB)),
         int((cosize(sA) + cosize(sB)) * 2 / 1024));
  for (int r : {0, 1, 2, 7, 8}) {
    printf("   r=%d:", r);
    for (int c : {0, 8, 16, 56, 63}) printf(" %4d", int(sA(r, c, 0)));
    printf("\n");
  }

  // ④ ⑤ MMA: 2x2 warps of m16n8k16, Tile<32,32,16>
  auto mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_2, _2>>{}, Tile<_32, _32, _16>{});
  auto cC = make_identity_tensor(make_shape(bM, bN));
  auto sAt = make_tensor(make_smem_ptr((bfloat16_t*)nullptr), sA(_, _, 0).layout());
  auto sBt = make_tensor(make_smem_ptr((bfloat16_t*)nullptr), sB(_, _, 0).layout());
  printf("④ fragment shapes per thread: A "); print(shape(mma.get_slice(0).partition_fragment_A(sAt)));
  printf("  B "); print(shape(mma.get_slice(0).partition_fragment_B(sBt))); printf("\n");
  printf("⑤ accumulators (C tile coordinates)\n");
  for (int t : {0, 1, 4, 32, 64, 127}) {
    auto p = mma.get_slice(t).partition_C(cC);
    printf("   t=%3d:", t);
    for (int i = 0; i < size(p); ++i) printf(" (%d,%d)", int(get<0>(p(i))), int(get<1>(p(i))));
    printf("\n");
  }
}
