// tile an (8,64) block to (128,64)
#include <cute/tensor.hpp>
using namespace cute;
int main() {
  auto r = tile_to_shape(Layout<Shape<_8,_64>,Stride<_64,_1>>{}, make_shape(_128{}, _64{}));
  print(r);
  return 0;
}
