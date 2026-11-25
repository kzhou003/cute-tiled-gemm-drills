// composition((4,6):(1,5), 3:6) -- 6 is neither a multiple of 4 nor < 4
#include <cute/tensor.hpp>
using namespace cute;
int main() {
  auto r = composition(Layout<Shape<_4,_6>,Stride<_1,_5>>{}, Layout<_3,_6>{});
  print(r);
  return 0;
}
