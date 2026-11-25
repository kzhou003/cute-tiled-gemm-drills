// D1 oracle: the SMEM layouts a TMA box lands in. CuTe's GMMA K-major swizzle atoms
// (Layout_K_SW32/64/128_Atom, mma_sm90_desc.hpp) tiled to (64, inner) half elements: for each
// colex index m + 64*k, the element offset inside the swizzled SMEM tile.
#include "common.hpp"
#include <cute/atom/mma_traits_sm90_gmma.hpp>
using namespace cute;

template <class Atom, int Inner>
S one(const char* mode, Atom) {
  auto L = tile_to_shape(Atom{}, Shape<_64, Int<Inner>>{});
  auto t = make_tensor(make_smem_ptr((half_t*)nullptr), L);
  std::vector<S> offs;
  for (int i = 0; i < 64 * Inner; ++i) offs.push_back(jnum(&t(i) - (half_t*)nullptr));
  return jobj({{"mode", jstr(mode)}, {"rows", jnum(64)}, {"inner", jnum(Inner)},
               {"atom", jstr(to_str(Atom{}))}, {"offsets", jarr(offs)}});
}

int main() {
  std::printf("%s\n", jobj({{"source", jstr("tile_to_shape(GMMA::Layout_K_SWxx_Atom<half_t>, (64, inner))")},
                            {"tiles", jarr({one<GMMA::Layout_K_SW32_Atom<half_t>, 16>("32B", {}),
                                            one<GMMA::Layout_K_SW64_Atom<half_t>, 32>("64B", {}),
                                            one<GMMA::Layout_K_SW128_Atom<half_t>, 64>("128B", {}),
                                            one<GMMA::Layout_K_SW128_Atom<half_t>, 128>("128B", {})})}}).c_str());
}
