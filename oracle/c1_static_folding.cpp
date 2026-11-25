// C1 oracle: the gmem offsets each thread copies in each k-tile, computed by real CuTe partition_S
// (static layouts), and the same with fully dynamic shapes/strides; plus the TV layout printout.
#include "common.hpp"
using namespace cute;

template <class GA>
S table(GA const& gA_layout) {
  auto copy = make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, half_t>{},
                              Layout<Shape<_32, _4>, Stride<_4, _1>>{}, Layout<Shape<_1, _4>>{});
  auto gA = make_tensor(make_gmem_ptr((half_t*)nullptr), gA_layout);
  std::vector<S> rows;   // rows[k][tid] = [4 offsets]
  for (int k = 0; k < 4; ++k) {
    auto tile = local_tile(gA, make_shape(_32{}, _16{}), make_coord(0, k));
    std::vector<S> per_t;
    for (int t = 0; t < 128; ++t) {
      auto part = copy.get_slice(t).partition_S(tile);
      std::vector<S> offs;
      for (int i = 0; i < int(size(part)); ++i) offs.push_back(jnum(&part(i) - (half_t*)nullptr));
      per_t.push_back(jarr(offs));
    }
    rows.push_back(jarr(per_t));
  }
  return jarr(rows);
}

int main() {
  auto copy = make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, half_t>{},
                              Layout<Shape<_32, _4>, Stride<_4, _1>>{}, Layout<Shape<_1, _4>>{});
  auto stat = make_layout(make_shape(_64{}, _64{}), make_stride(_64{}, _1{}));
  int M = 64, K = 64;   // runtime values: the same tensor with dynamic shape and stride
  auto dyn = make_layout(make_shape(M, K), make_stride(K, 1));
  std::printf("%s\n", jobj({{"source", jstr("TiledCopy (32,4):(4,1) x (1,4), uint64 atom, partition_S of local_tile")},
                            {"tv_layout", jstr(to_str(copy.get_layoutS_TV()))},
                            {"dynamic_layout", jstr(to_str(dyn))},
                            {"static_offsets", table(stat)}, {"dynamic_offsets", table(dyn)}}).c_str());
}
