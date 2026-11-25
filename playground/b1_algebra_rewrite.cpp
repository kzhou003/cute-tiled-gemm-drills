// B1: the three Track-A mappings built with Layout Algebra.
#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;

int main() {
  auto gA = make_layout(make_shape(_64{}, _64{}), make_stride(_64{}, _1{}));
  auto z = zipped_divide(gA, make_shape(_32{}, _16{}));
  print("zipped_divide(gA, (32,16)) = "); print(z); print("\n");
  printf("rest mode at (1,2) = %d   (the tile origin of CTA row 1, k-tile 2)\n", int(layout<1>(z)(make_coord(1, 2))));
  auto thr = Layout<Shape<_32, _4>, Stride<_4, _1>>{};
  auto val = Layout<Shape<_1, _4>>{};
  auto mn = raked_product(thr, val);
  print("raked_product(thr, val)   = "); print(mn); print("\n");
  print("right_inverse(...)         = "); print(right_inverse(mn)); print("\n");
  auto tv = right_inverse(mn).with_shape(make_shape(size(thr), size(val)));
  print("layout_tv                  = "); print(tv); print("\n");
  printf("tv(5, 0..3) = %d %d %d %d  (column-major index m + 32k in the 32x16 tile)\n",
         int(tv(5, 0)), int(tv(5, 1)), int(tv(5, 2)), int(tv(5, 3)));
  auto sm = composition(Swizzle<2, 2, 2>{}, Layout<Shape<_32, _16>, Stride<_16, _1>>{});
  print("smem                       = "); print(sm); print("\n");
}
