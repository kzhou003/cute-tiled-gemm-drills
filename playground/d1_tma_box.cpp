// D1: where a TMA box lands in SMEM. The GMMA K-major SW128 atom on a real smem pointer.
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cstdio>
using namespace cute;

int main() {
  using Atom = GMMA::Layout_K_SW128_Atom<half_t>;
  print("K_SW128 half atom : "); print(Atom{}); print("\n");
  auto t = make_tensor(make_smem_ptr((half_t*)nullptr), tile_to_shape(Atom{}, Shape<_64, _64>{}));
  printf("physical element offset of (m, k) for k = 0, 8, ..., 56:\n");
  for (int m : {0, 1, 2, 7}) {
    printf(" m=%d:", m);
    for (int k = 0; k < 64; k += 8) printf(" %5ld", long(&t(m, k) - (half_t*)nullptr));
    printf("\n");
  }
  int agree = 0;
  for (int m = 0; m < 64; ++m)
    for (int k = 0; k < 64; ++k) {
      int o = m * 64 + k;
      agree += (&t(m, k) - (half_t*)nullptr) == (o ^ ((o >> 3) & 0x38));
    }
  printf("element-domain Swizzle<3,3,3> agrees on %d of 4096 elements\n", agree);
}
