// A4: the SM80 m16n8k16 atom and a 2x2-warp TiledMMA. Who holds A(9,2) and C(5,7)?
#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;

int main() {
  using Tr = MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>;
  print("atom A TV: "); print(typename Tr::ALayout{}); print("\n");
  print("atom B TV: "); print(typename Tr::BLayout{}); print("\n");
  print("atom C TV: "); print(typename Tr::CLayout{}); print("\n");
  auto mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_2, _2, _1>>{}, Tile<_32, _32, _16>{});
  auto tvA = mma.get_layoutA_TV();
  auto tvC = mma.get_layoutC_TV();
  print("tiled A TV: "); print(tvA); print("\n");
  for (int t = 0; t < 128; ++t)
    for (int v = 0; v < 8; ++v) {
      if (tvA(t, v) == 9 + 32 * 2) printf("A(9,2) is value %d of thread %d\n", v, t);
      if (tvC(t, v) == 5 + 32 * 7) printf("C(5,7) is value %d of thread %d\n", v, t);
    }
  auto sA = make_tensor<half_t>(make_shape(_32{}, _16{}));
  print("partition_fragment_A shape: "); print(shape(mma.get_slice(0).partition_fragment_A(sA))); print("\n");
}
