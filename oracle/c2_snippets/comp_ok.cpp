// composition((4,8):(8,1), (2,4):(1,2)) -- legal
#include <cute/tensor.hpp>
using namespace cute;
int main() {
  auto r = composition(Layout<Shape<_4,_8>,Stride<_8,_1>>{}, Layout<Shape<_2,_4>,Stride<_1,_2>>{});
  print(r);
  return 0;
}
