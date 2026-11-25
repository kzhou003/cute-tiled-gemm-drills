// composition((4,6):(1,5), 3:d) with a runtime stride -- cannot be checked statically
#include <cute/tensor.hpp>
using namespace cute;
int main() {
  int d = 6;
  auto r = composition(Layout<Shape<_4,_6>,Stride<_1,_5>>{}, make_layout(_3{}, d));
  print(r);
  return 0;
}
