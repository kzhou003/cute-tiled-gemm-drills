// composition((4,6):(1,4), 3:6) -- lhs coalesces to 24:1, so it is legal
#include <cute/tensor.hpp>
using namespace cute;
int main() {
  auto r = composition(Layout<Shape<_4,_6>,Stride<_1,_4>>{}, Layout<_3,_6>{});
  print(r);
  return 0;
}
