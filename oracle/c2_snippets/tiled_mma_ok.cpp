// partition_C of a divisible 32x32 tile
#include <cute/tensor.hpp>
using namespace cute;
int main() {
  auto mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_2,_2,_1>>{}, Tile<_32,_32,_16>{});
  auto C = make_tensor<float>(make_shape(_32{}, _32{}));
  print(mma.get_slice(0).partition_C(C).layout());
  return 0;
}
