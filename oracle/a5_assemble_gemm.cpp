// A5 oracle: per-thread partitions of the assembled 32x32x16 CTA tile with real CuTe, the way
// sgemm_sm80.cu builds them: TiledCopy (32,4):(4,1) x (1,4) partition_S on the A/B tile,
// TiledMMA (2,2,1) Tile<32,32,16> partition_A/B/C, and the Swizzle<2,2,2> SMEM offsets.
#include "common.hpp"
using namespace cute;

template <class T>
S coords_of(T const& part) {
  std::vector<S> cs;
  for (int i = 0; i < int(size(part)); ++i) cs.push_back(jpair(get<0>(part(i)), get<1>(part(i))));
  return jarr(cs);
}

int main() {
  auto copy = make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, half_t>{},
                              Layout<Shape<_32, _4>, Stride<_4, _1>>{}, Layout<Shape<_1, _4>>{});
  auto mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_2, _2, _1>>{}, Tile<_32, _32, _16>{});
  auto idAB = make_identity_tensor(make_shape(_32{}, _16{}));
  auto idC = make_identity_tensor(make_shape(_32{}, _32{}));
  std::vector<S> cp, pa, pb, pc;
  for (int t = 0; t < 128; ++t) {
    cp.push_back(coords_of(copy.get_slice(t).partition_S(idAB)));
    auto thr = mma.get_slice(t);
    pa.push_back(coords_of(thr.partition_A(idAB)));
    pb.push_back(coords_of(thr.partition_B(idAB)));
    pc.push_back(coords_of(thr.partition_C(idC)));
  }
  auto sw = composition(Swizzle<2, 2, 2>{}, Layout<Shape<_32, _16>, Stride<_16, _1>>{});
  std::vector<S> rows;
  for (int r = 0; r < 32; ++r) {
    std::vector<S> row;
    for (int c = 0; c < 16; ++c) row.push_back(jnum(sw(r, c)));
    rows.push_back(jarr(row));
  }
  std::printf("%s\n", jobj({{"source", jstr("TiledCopy/TiledMMA partitions of identity tensors (sgemm_sm80 structure)")},
                            {"copy", jarr(cp)}, {"mma_A", jarr(pa)}, {"mma_B", jarr(pb)}, {"mma_C", jarr(pc)},
                            {"smem_sw222", jarr(rows)}}).c_str());
}
