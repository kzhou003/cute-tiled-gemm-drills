// A5: the sgemm_sm80 structure in one place: per-thread partitions of one CTA tile.
#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;

int main() {
  auto copy = make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, half_t>{},
                              Layout<Shape<_32, _4>, Stride<_4, _1>>{}, Layout<Shape<_1, _4>>{});
  auto mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_2, _2, _1>>{}, Tile<_32, _32, _16>{});
  auto sA = composition(Swizzle<2, 2, 2>{}, Layout<Shape<_32, _16>, Stride<_16, _1>>{});
  auto idA = make_identity_tensor(make_shape(_32{}, _16{}));
  auto idC = make_identity_tensor(make_shape(_32{}, _32{}));
  for (int t : {20, 21, 85}) {
    auto cp = copy.get_slice(t).partition_S(idA);
    auto pa = mma.get_slice(t).partition_A(idA);
    printf("thread %3d copies", t);
    for (int i = 0; i < size(cp); ++i) { print(" "); print(cp(i)); }
    printf("\n           holds A");
    for (int i = 0; i < size(pa); ++i) { print(" "); print(pa(i)); }
    printf("\n");
  }
  auto pc = mma.get_slice(23).partition_C(idC);
  printf("thread  23 accumulates C");
  for (int i = 0; i < size(pc); ++i) { print(" "); print(pc(i)); }
  printf("\nA(5,2) lives at smem offset %d (Swizzle<2,2,2>)\n", int(sA(5, 2)));
}
