// E1: the SMEM side of the Hopper GEMM: a 128x64 SW128 stage, its offsets, and the stage byte count
// the producer declares with arrive_and_expect_tx.
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cstdio>
using namespace cute;

int main() {
  using Atom = GMMA::Layout_K_SW128_Atom<half_t>;
  auto sA_layout = tile_to_shape(Atom{}, Shape<_128, _64>{});
  print("SW128 atom  : "); print(Atom{}); print("\n");
  print("sA (128x64) : "); print(sA_layout); print("\n");
  auto sA = make_tensor(make_smem_ptr((half_t*)nullptr), sA_layout);
  auto off = [&](int m, int k) { return long(&sA(m, k) - (half_t*)nullptr); };
  printf("offset(0,8) = %ld offset(1,0) = %ld offset(1,8) = %ld offset(7,0) = %ld offset(8,0) = %ld\n",
         off(0, 8), off(1, 0), off(1, 8), off(7, 0), off(8, 0));
  int bad = 0;
  for (int m = 0; m < 128; ++m)
    for (int k = 0; k < 64; ++k) {
      long o = m * 64 + k;
      bad += off(m, k) != (o ^ ((o >> 3) & 0x38));
    }
  printf("sw128(m*64+k) vs real CuTe offset: %d mismatches of %d\n", bad, 128 * 64);
  printf("stage bytes (A 128x64 + B 128x64, half) = %d\n", int(2 * 128 * 64 * sizeof(half_t)));
}
