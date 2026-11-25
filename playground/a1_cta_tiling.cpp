// A1: local_tile is a view. Print the C / A views of a few CTAs and the elem_less predicate
// for the ragged 70x50 case.
#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;

int main() {
  auto cta_tiler = make_shape(Int<32>{}, Int<32>{}, Int<16>{});    // (BM, BN, BK)
  auto mC = make_identity_tensor(make_shape(70, 50));               // ragged on purpose
  for (int pm = 0; pm < 3; ++pm) {
    for (int pn = 0; pn < 2; ++pn) {
      auto gC = local_tile(mC, select<0, 1>(cta_tiler), make_coord(pm, pn));
      int live = 0;
      for (int i = 0; i < size(gC); ++i) live += elem_less(gC(i), shape(mC));
      printf("CTA (%d,%d): live %4d / %d\n", pm, pn, live, int(size(gC)));
      print("   gC(0) = "); print(gC(0)); print("  gC(last) = "); print(gC(size(gC) - 1)); print("\n");
    }
  }
  auto mA = make_identity_tensor(make_shape(70, 40));
  auto gA = local_tile(mA, select<0, 2>(cta_tiler), make_coord(1, _));  // all K tiles of row-block 1
  print("gA(pid_m=1, all k): "); print(gA.layout()); print("\n");
  auto real = make_tensor(make_gmem_ptr((float*)nullptr), make_layout(make_shape(64, 64), make_stride(64, Int<1>{})));
  auto v = local_tile(real, make_shape(Int<32>{}, Int<32>{}), make_coord(0, 1));
  printf("view of CTA (0,1) on row-major 64x64: offset(0,0)=%ld offset(1,0)=%ld\n",
         long(&v(0, 0) - (float*)nullptr), long(&v(1, 0) - (float*)nullptr));
}
