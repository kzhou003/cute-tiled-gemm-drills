// local_tile of a divisible tile
#include <cute/tensor.hpp>
using namespace cute;
int main() {
  auto A = make_identity_tensor(make_shape(_64{}, _64{}));
  print(local_tile(A, make_shape(_32{}, _16{}), make_coord(1, 2)).layout());
  return 0;
}
