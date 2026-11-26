// F4 oracle: what the identity-tensor partitions of solutions/f4_epilogue_residue.cu say for a
// ragged 1000 x 760 problem (Config<128,128,64,3,2,4>):
//   load predicates   get<0>(tAcA(0, i, 0, 0)) < M   for a few (bm, t)
//   store counts      how many accumulators pass elem_less(tCcC(i), (M, N)), summed per CTA
#include "common.hpp"
#include "tiled_gemm_config.hpp"
using namespace cute;

using Cfg = cfg::Config<128, 128, 64, 3, 2, 4>;

int main() {
  int M = 1000, N = 760, K = 128;
  auto cA = make_identity_tensor(make_shape(M, K));
  auto cC = make_identity_tensor(make_shape(M, N));
  auto tiler = make_shape(Int<Cfg::BM>{}, Int<Cfg::BN>{}, Int<Cfg::BK>{});
  typename Cfg::G2SCopy g2s;
  typename Cfg::TiledMma mma;
  int gm = (M + Cfg::BM - 1) / Cfg::BM, gn = (N + Cfg::BN - 1) / Cfg::BN;
  std::vector<S> preds;
  for (int bm : {0, gm - 1}) {
    for (int t : {0, 13, 100, 255}) {
      auto gcA = local_tile(cA, tiler, make_coord(bm, 0, _), Step<_1, X, _1>{});
      auto tAcA = g2s.get_slice(t).partition_S(gcA);
      std::vector<S> p;
      for (int i = 0; i < int(size<1>(tAcA)); ++i) p.push_back(get<0>(tAcA(0, i, 0, 0)) < M ? "true" : "false");
      preds.push_back(jobj({{"bm", jnum(bm)}, {"t", jnum(t)}, {"pred", jarr(p)}}));
    }
  }
  std::vector<S> counts;
  long long total = 0;
  for (int bm = 0; bm < gm; ++bm)
    for (int bn = 0; bn < gn; ++bn) {
      auto gcC = local_tile(cC, tiler, make_coord(bm, bn, _), Step<_1, _1, X>{});
      long long live = 0;
      for (int t = 0; t < Cfg::Threads; ++t) {
        auto tCcC = mma.get_slice(t).partition_C(gcC);
        for (int i = 0; i < int(size(tCcC)); ++i) live += elem_less(tCcC(i), make_shape(M, N)) ? 1 : 0;
      }
      total += live;
      counts.push_back(jobj({{"bm", jnum(bm)}, {"bn", jnum(bn)}, {"live", jnum(live)}}));
    }
  std::printf("%s\n", jobj({{"source", jstr("identity-tensor partitions of Config<128,128,64,3,2,4>")},
                            {"M", jnum(M)}, {"N", jnum(N)}, {"grid", jpair(gm, gn)},
                            {"load_predicates", jarr(preds)}, {"store_counts", jarr(counts)},
                            {"store_total", jnum(total)}}).c_str());
}
