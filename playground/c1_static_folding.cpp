// C1: the offsets a thread copies, computed by real CuTe. Compare with oracle/c1_asm.sh, which
// shows what the compiler turns the static and the dynamic versions into.
#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;

int main() {
  auto copy = make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, half_t>{},
                              Layout<Shape<_32, _4>, Stride<_4, _1>>{}, Layout<Shape<_1, _4>>{});
  print("TV layout (tid,v) -> tile idx : "); print(copy.get_layoutS_TV()); print("\n");
  auto gA = make_tensor(make_gmem_ptr((half_t*)nullptr), Layout<Shape<_64, _64>, Stride<_64, _1>>{});
  auto tile = local_tile(gA, make_shape(_32{}, _16{}), make_coord(0, 1));   // k = 1
  for (int t : {0, 1, 4, 127}) {
    auto p = copy.get_slice(t).partition_S(tile);
    printf("tid %3d:", t);
    for (int i = 0; i < size(p); ++i) printf(" %5ld", long(&p(i) - (half_t*)nullptr));
    printf("\n");
  }
}
