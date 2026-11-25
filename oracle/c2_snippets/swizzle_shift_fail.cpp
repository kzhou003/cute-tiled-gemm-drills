// Swizzle<3,3,2> -- |S| < B
#include <cute/tensor.hpp>
using namespace cute;
int main() {
  auto s = Swizzle<3,3,2>{};
  print(s(17));
  return 0;
}
