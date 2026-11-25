// A2 oracle: real make_tiled_copy(atom, ThrLayout, ValLayout) for four ways of splitting a
// 32x16 half tile over 128 threads. Prints get_layoutS_TV() (string + full value table) and,
// for every thread, the tile coordinates of partition_S(identity tensor).
#include "common.hpp"
using namespace cute;

template <class Thr, class Val>
S variant(const char* name, Thr thr, Val val) {
  auto tc = make_tiled_copy(Copy_Atom<UniversalCopy<half_t>, half_t>{}, thr, val);
  auto tv = tc.get_layoutS_TV();
  auto id = make_identity_tensor(make_shape(Int<32>{}, Int<16>{}));
  std::vector<S> threads;
  for (int t = 0; t < int(size(thr)); ++t) {
    auto part = tc.get_slice(t).partition_S(id);
    std::vector<S> cs;
    for (int i = 0; i < int(size(part)); ++i) cs.push_back(jpair(get<0>(part(i)), get<1>(part(i))));
    threads.push_back(jarr(cs));
  }
  return jobj({{"name", jstr(name)}, {"thr", jstr(to_str(thr))}, {"val", jstr(to_str(val))},
               {"tiler", jstr(to_str(typename decltype(tc)::Tiler_MN{}))},
               {"tv_layout", jstr(to_str(tv))}, {"tv_values", values(tv)}, {"threads", jarr(threads)}});
}

int main() {
  std::vector<S> vs = {
      variant("base", Layout<Shape<_32, _4>, Stride<_4, _1>>{}, Layout<Shape<_1, _4>>{}),
      variant("colthr", Layout<Shape<_32, _4>, Stride<_1, _32>>{}, Layout<Shape<_1, _4>>{}),
      variant("t8x16_v4x1", Layout<Shape<_8, _16>, Stride<_16, _1>>{}, Layout<Shape<_4, _1>>{}),
      variant("t16x8_v2x2", Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_2, _2>>{}),
  };
  std::printf("%s\n", jobj({{"source", jstr("cute::make_tiled_copy(...).get_layoutS_TV() / partition_S")},
                            {"variants", jarr(vs)}}).c_str());
}
