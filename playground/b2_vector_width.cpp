// B2: max_common_vector / max_alignment against a gmem tile (32,16):(64,1).
#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;

template <class L>
void row(const char* name, L l) {
  auto g = Layout<Shape<_32, _16>, Stride<_64, _1>>{};
  printf("%-10s mcv = %3d   max_alignment(smem) = %3d   mcl = ", name, int(max_common_vector(g, l)),
         int(max_alignment(l)));
  print(max_common_layout(g, l));
  print("\n");
}

int main() {
  auto s = Layout<Shape<_32, _16>, Stride<_16, _1>>{};
  row("plain", s);
  row("Sw<1,1,1>", composition(Swizzle<1, 1, 1>{}, s));
  row("Sw<2,2,2>", composition(Swizzle<2, 2, 2>{}, s));
  row("Sw<3,3,3>", composition(Swizzle<3, 3, 3>{}, s));
  row("pad 18", Layout<Shape<_32, _16>, Stride<Int<18>, _1>>{});
  row("M-major", Layout<Shape<_32, _16>, Stride<_1, _32>>{});
}
