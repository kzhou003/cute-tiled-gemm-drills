// A2: make_tiled_copy(atom, ThrLayout, ValLayout) and what each thread copies.
#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;

int main() {
  auto copy = make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, half_t>{},
                              Layout<Shape<_32, _4>, Stride<_4, _1>>{},    // ThrLayout
                              Layout<Shape<_1, _4>>{});                    // ValLayout
  print("tiler     : "); print(typename decltype(copy)::Tiler_MN{}); print("\n");
  print("TV layout : "); print(copy.get_layoutS_TV()); print("\n");
  auto id = make_identity_tensor(make_shape(_32{}, _16{}));
  for (int t : {0, 1, 4, 5, 127}) {
    auto p = copy.get_slice(t).partition_S(id);
    printf("thread %3d :", t);
    for (int i = 0; i < size(p); ++i) { print(" "); print(p(i)); }
    print("\n");
  }
}
