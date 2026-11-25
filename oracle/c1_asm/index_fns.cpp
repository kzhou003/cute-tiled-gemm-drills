// Two ways to compute the same copy offset: with static CuTe layouts (Int<N>) and with runtime
// shapes/strides. oracle/c1_asm.sh compiles this with -O2 -S and counts multiply/divide
// instructions in each function.
#include <cute/tensor.hpp>
using namespace cute;

extern "C" int offset_static(int tid, int v, int k) {
  auto tv = Layout<Shape<Shape<_4, _32>, Shape<_4, _1>>, Stride<Stride<_128, _1>, Stride<_32, _0>>>{};
  auto tile = Layout<Shape<_32, _16>>{};                       // tile index -> (m, kk), colex
  auto gmem = Layout<Shape<_64, _64>, Stride<_64, _1>>{};
  auto mk = tile.get_hier_coord(tv(make_coord(tid, v)));
  return gmem(get<0>(mk), k * 16 + get<1>(mk));
}

extern "C" int offset_dynamic(int tid, int v, int k, int bm, int bk, int ld) {
  auto tv = make_layout(make_shape(make_shape(4, 32), make_shape(4, 1)),
                        make_stride(make_stride(4 * bm, 1), make_stride(bm, 0)));
  auto tile = make_layout(make_shape(bm, bk));
  auto gmem = make_layout(make_shape(2 * bm, 4 * bk), make_stride(ld, 1));
  auto mk = tile.get_hier_coord(tv(make_coord(tid, v)));
  return gmem(get<0>(mk), k * bk + get<1>(mk));
}
