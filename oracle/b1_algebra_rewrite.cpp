// B1 oracle: the three Track-A mappings built with real CuTe Layout Algebra, plus a few
// core-algebra cases that pin the given Python mirror to CuTe's exact printed results.
#include "common.hpp"
using namespace cute;

template <class Thr, class Val>
S tv_case(const char* name, Thr thr, Val val) {
  auto mn = raked_product(thr, val);
  auto inv = right_inverse(mn);
  auto tv = inv.with_shape(make_shape(size(thr), size(val)));
  return jobj({{"name", jstr(name)}, {"layout_mn", jstr(to_str(mn))}, {"right_inverse", jstr(to_str(inv))},
               {"layout_tv", jstr(to_str(tv))}, {"tv_values", values(tv)},
               {"tiler", jstr(to_str(product_each(shape(mn))))}});
}

int main() {
  auto gA = make_layout(make_shape(_64{}, _64{}), make_stride(_64{}, _1{}));
  auto z = zipped_divide(gA, make_shape(_32{}, _16{}));
  std::vector<S> rest;
  for (int bm = 0; bm < 2; ++bm)
    for (int k = 0; k < 4; ++k)
      rest.push_back(jobj({{"coord", jpair(bm, k)}, {"offset", jnum(layout<1>(z)(make_coord(bm, k)))}}));
  // the slice the real local_tile produces on a tensor whose data pointer starts at 0
  auto gT = make_tensor(make_gmem_ptr((half_t*)nullptr), gA);
  auto lt = local_tile(gT, make_shape(_32{}, _16{}), make_coord(1, 2));
  std::vector<S> core = {
      jobj({{"expr", jstr("composition((4,8):(8,1), 8:2)")},
            {"got", jstr(to_str(composition(Layout<Shape<_4, _8>, Stride<_8, _1>>{}, Layout<_8, _2>{})))}}),
      jobj({{"expr", jstr("complement(4:2, 24)")}, {"got", jstr(to_str(complement(Layout<_4, _2>{}, _24{})))}}),
      jobj({{"expr", jstr("coalesce((2,(1,6)):(1,(6,2)))")},
            {"got", jstr(to_str(coalesce(make_layout(make_shape(_2{}, make_shape(_1{}, _6{})),
                                                     make_stride(_1{}, make_stride(_6{}, _2{}))))))}}),
      jobj({{"expr", jstr("blocked_product(4:1, (2,3):(1,2))")},
            {"got", jstr(to_str(blocked_product(Layout<_4>{}, Layout<Shape<_2, _3>>{})))}}),
      jobj({{"expr", jstr("raked_product(4:1, (2,3):(1,2))")},
            {"got", jstr(to_str(raked_product(Layout<_4>{}, Layout<Shape<_2, _3>>{})))}}),
      jobj({{"expr", jstr("logical_divide(24:1, 4:2)")},
            {"got", jstr(to_str(logical_divide(Layout<_24>{}, Layout<_4, _2>{})))}}),
      jobj({{"expr", jstr("right_inverse((4,8):(8,1))")},
            {"got", jstr(to_str(right_inverse(Layout<Shape<_4, _8>, Stride<_8, _1>>{})))}}),
      jobj({{"expr", jstr("zipped_divide((8,6):(1,8), (4,3))")},
            {"got", jstr(to_str(zipped_divide(Layout<Shape<_8, _6>>{}, make_shape(_4{}, _3{}))))}}),
  };
  std::vector<S> tvs = {
      tv_case("base", Layout<Shape<_32, _4>, Stride<_4, _1>>{}, Layout<Shape<_1, _4>>{}),
      tv_case("colthr", Layout<Shape<_32, _4>, Stride<_1, _32>>{}, Layout<Shape<_1, _4>>{}),
      tv_case("t8x16_v4x1", Layout<Shape<_8, _16>, Stride<_16, _1>>{}, Layout<Shape<_4, _1>>{}),
      tv_case("t16x8_v2x2", Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_2, _2>>{}),
  };
  auto sm = composition(Swizzle<2, 2, 2>{}, Layout<Shape<_32, _16>, Stride<_16, _1>>{});
  std::printf("%s\n", jobj({{"source", jstr("cute Layout Algebra at the pinned CUTLASS commit")},
                            {"zipped_divide", jstr(to_str(z))}, {"rest", jarr(rest)},
                            {"local_tile_1_2", jobj({{"layout", jstr(to_str(lt.layout()))},
                                                     {"offset", jnum(lt.data().get() - (half_t*)nullptr)}})},
                            {"tv", jarr(tvs)}, {"smem", values(sm)}, {"core", jarr(core)}}).c_str());
}
