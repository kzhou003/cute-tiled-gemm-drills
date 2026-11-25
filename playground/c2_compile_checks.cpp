// C2: things CuTe checks at compile time. Uncomment one of the lines below and run again to see the
// static_assert message (oracle/c2_try_compile.sh does this for every file in oracle/c2_snippets/).
#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;

int main() {
  print(composition(Layout<Shape<_4, _6>, Stride<_1, _4>>{}, Layout<_3, _6>{})); print("   ok: lhs coalesces to 24:1\n");
  // print(composition(Layout<Shape<_4, _6>, Stride<_1, _5>>{}, Layout<_3, _6>{}));   // Stride Divisibility Condition
  // print(composition(Layout<Shape<_4, _6>, Stride<_1, _5>>{}, Layout<_3, _2>{}));   // Shape Divisibility Condition
  // auto s = Swizzle<3, 3, 2>{};                                                       // abs(SShift) must be more than BBits.
  // auto c = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, half_t>{},          // too few vals for selected CopyAtom
  //                          Layout<Shape<_32, _4>, Stride<_4, _1>>{}, Layout<Shape<_1, _4>>{});
  int d = 6;   // a runtime stride: the divisibility check is skipped
  print(composition(Layout<Shape<_4, _6>, Stride<_1, _5>>{}, make_layout(_3{}, d))); print("   compiles: dynamic\n");
}
