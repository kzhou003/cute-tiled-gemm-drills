// local_tile allows a ragged last tile
#include <cute/tensor.hpp>
using namespace cute;
int main() {
  auto A = make_identity_tensor(make_shape(70, 50));
  print(local_tile(A, make_shape(_32{}, _32{}), make_coord(2, 1)).layout());
  return 0;
}
