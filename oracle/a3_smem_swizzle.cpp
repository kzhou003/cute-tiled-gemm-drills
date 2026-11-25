// A3 oracle: real CuTe composition(Swizzle<B,M,S>{}, (32,16):(16,1)). For each swizzle print the
// ComposedLayout and its 512 offsets in colex order (index i = m + 32*k).
#include "common.hpp"
using namespace cute;

using Plain = Layout<Shape<_32, _16>, Stride<_16, _1>>;

template <int B, int M, int Sh>
S one() {
  auto L = composition(Swizzle<B, M, Sh>{}, Plain{});
  return jobj({{"sw", "[" + jnum(B) + "," + jnum(M) + "," + jnum(Sh) + "]"}, {"layout", jstr(to_str(L))},
               {"offsets", values(L)}});
}

int main() {
  std::vector<S> v = {one<2, 2, 2>(), one<1, 3, 3>(), one<2, 3, 3>(), one<3, 3, 3>(),
                      one<2, 2, 4>(), one<1, 2, 3>(), one<2, 1, 3>()};
  std::printf("%s\n", jobj({{"source", jstr("cute::composition(Swizzle<B,M,S>{}, (32,16):(16,1))")},
                            {"plain", jobj({{"layout", jstr(to_str(Plain{}))}, {"offsets", values(Plain{})}})},
                            {"swizzled", jarr(v)}}).c_str());
}
