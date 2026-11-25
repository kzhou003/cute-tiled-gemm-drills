// A4 oracle: the SM80 m16n8k16 atom's A/B/C TV layouts (MMA_Traits) and the 2x2-warp TiledMMA
// make_tiled_mma(atom, Layout<(2,2,1)>, Tile<32,32,16>): get_layoutA/B/C_TV value tables and the
// shapes of partition_fragment_A/B/C for one thread.
#include "common.hpp"
using namespace cute;

int main() {
  using Atom = SM80_16x8x16_F32F16F16F32_TN;
  using Tr = MMA_Traits<Atom>;
  auto mma = make_tiled_mma(Atom{}, Layout<Shape<_2, _2, _1>>{}, Tile<_32, _32, _16>{});
  auto thr = mma.get_slice(0);
  auto sA = make_tensor(make_smem_ptr((half_t*)nullptr), make_layout(make_shape(_32{}, _16{})));
  auto sB = make_tensor(make_smem_ptr((half_t*)nullptr), make_layout(make_shape(_32{}, _16{})));
  auto gC = make_tensor(make_gmem_ptr((float*)nullptr), make_layout(make_shape(_32{}, _32{})));
  auto fA = thr.partition_fragment_A(sA);
  auto fB = thr.partition_fragment_B(sB);
  auto fC = thr.partition_fragment_C(gC);
  std::printf("%s\n", jobj({
      {"source", jstr("MMA_Traits<SM80_16x8x16_F32F16F16F32_TN> + make_tiled_mma(atom, (2,2,1), Tile<32,32,16>)")},
      {"atom", jobj({{"A_layout", jstr(to_str(typename Tr::ALayout{}))}, {"A", values(typename Tr::ALayout{})},
                     {"B_layout", jstr(to_str(typename Tr::BLayout{}))}, {"B", values(typename Tr::BLayout{})},
                     {"C_layout", jstr(to_str(typename Tr::CLayout{}))}, {"C", values(typename Tr::CLayout{})}})},
      {"tiled", jobj({{"A_layout", jstr(to_str(mma.get_layoutA_TV()))}, {"A", values(mma.get_layoutA_TV())},
                      {"B_layout", jstr(to_str(mma.get_layoutB_TV()))}, {"B", values(mma.get_layoutB_TV())},
                      {"C_layout", jstr(to_str(mma.get_layoutC_TV()))}, {"C", values(mma.get_layoutC_TV())}})},
      {"fragment_shape", jobj({{"A", jstr(to_str(shape(fA)))}, {"B", jstr(to_str(shape(fB)))},
                               {"C", jstr(to_str(shape(fC)))}})},
  }).c_str());
}
