// A3: Swizzle<B,M,S> o (32,16):(16,1). Print the first rows and check that it is a bijection.
#include <cute/tensor.hpp>
#include <cstdio>
#include <vector>
using namespace cute;

template <class L>
void show(const char* name, L const& L_) {
  print(name); print(" = "); print(L_); print("\n");
  std::vector<int> seen(512, 0);
  for (int i = 0; i < 512; ++i) seen[L_(i)]++;
  bool bij = true;
  for (int x : seen) bij &= (x == 1);
  for (int r = 0; r < 4; ++r) {
    printf("  row %d:", r);
    for (int c = 0; c < 16; ++c) printf(" %3d", int(L_(r, c)));
    printf("\n");
  }
  printf("  bijection: %s\n", bij ? "yes" : "no");
}

int main() {
  auto base = Layout<Shape<_32, _16>, Stride<_16, _1>>{};
  show("plain    ", base);
  show("Sw<2,2,2>", composition(Swizzle<2, 2, 2>{}, base));
  show("Sw<1,3,3>", composition(Swizzle<1, 3, 3>{}, base));
}
