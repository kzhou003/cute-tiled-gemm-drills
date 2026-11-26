// F2 oracle: the exact types of gpu/tiled_gemm_config.hpp (Config<128,128,64,1,2,4>) partitioning
// identity tensors on the host:
//   G2SCopy.get_slice(t).partition_S(cA)    the 32 (m,k) of thread t's four 16 B cp.async
//   SmemLayoutA(r, k, 0)                    the element offset of every (r, k) in the swizzled tile
//   TiledMma.get_slice(t).partition_C(cC)   the 64 (m,n) accumulators of thread t
#include "common.hpp"
#include "tiled_gemm_config.hpp"
using namespace cute;

using Cfg = cfg::Config<128, 128, 64, 1, 2, 4>;

template <class T>
S coords(T const& t) {
  std::vector<S> v;
  for (int i = 0; i < int(size(t)); ++i) v.push_back(jpair(get<0>(t(i)), get<1>(t(i))));
  return jarr(v);
}

int main() {
  auto cA = make_identity_tensor(make_shape(Int<Cfg::BM>{}, Int<Cfg::BK>{}));
  auto cC = make_identity_tensor(make_shape(Int<Cfg::BM>{}, Int<Cfg::BN>{}));
  typename Cfg::G2SCopy g2s;
  typename Cfg::TiledMma mma;
  std::vector<std::pair<S, S>> copy, mmac;
  for (int t : {0, 1, 7, 8, 37, 100, 200, 255}) {
    auto p = g2s.get_slice(t).partition_S(cA);
    copy.push_back({jnum(t), coords(p)});
  }
  for (int t : {0, 1, 2, 3, 4, 31, 32, 37, 255}) {
    auto p = mma.get_slice(t).partition_C(cC);
    mmac.push_back({jnum(t), coords(p)});
  }
  std::vector<S> rows;
  for (int r = 0; r < Cfg::BM; ++r) {
    std::vector<S> row;
    for (int k = 0; k < Cfg::BK; ++k) row.push_back(jnum(typename Cfg::SmemLayoutA{}(r, k, 0)));
    rows.push_back(jarr(row));
  }
  auto p0 = g2s.get_slice(0).partition_S(cA);
  auto c0 = mma.get_slice(0).partition_C(cC);
  std::printf("%s\n", jobj({{"source", jstr("cfg::Config<128,128,64,1,2,4> on identity tensors")},
                            {"smem_layout", jstr(to_str(typename Cfg::SmemLayoutA{}))},
                            {"copy_shape", jstr(to_str(shape(p0)))},
                            {"mma_c_shape", jstr(to_str(shape(c0)))},
                            {"copy", jobj(copy)},
                            {"mma_c", jobj(mmac)},
                            {"smem_offset", jarr(rows)},
                            {"smem_bytes", jnum(Cfg::SmemBytes)}}).c_str());
}
