// (8,64) block does not divide (12,64)
#include <cute/tensor.hpp>
using namespace cute;
int main() {
  auto r = tile_to_shape(Layout<Shape<_8,_64>,Stride<_64,_1>>{}, make_shape(_12{}, _64{}));
  print(r);
  return 0;
}
