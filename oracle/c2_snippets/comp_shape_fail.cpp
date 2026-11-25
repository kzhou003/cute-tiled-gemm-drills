// composition((4,6):(1,5), 3:2) -- 3 elements cannot be split evenly
#include <cute/tensor.hpp>
using namespace cute;
int main() {
  auto r = composition(Layout<Shape<_4,_6>,Stride<_1,_5>>{}, Layout<_3,_2>{});
  print(r);
  return 0;
}
