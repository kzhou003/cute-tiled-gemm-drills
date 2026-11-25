// Swizzle<2,2,2> -- legal
#include <cute/tensor.hpp>
using namespace cute;
int main() {
  auto s = Swizzle<2,2,2>{};
  print(s(17));
  return 0;
}
