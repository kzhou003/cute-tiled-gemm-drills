// 128-bit atom needs 8 half values per thread, ValLayout gives 4
#include <cute/tensor.hpp>
using namespace cute;
int main() {
  auto c = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, half_t>{},
                           Layout<Shape<_32,_4>,Stride<_4,_1>>{}, Layout<Shape<_1,_4>>{});
  print(c);
  return 0;
}
