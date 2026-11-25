// A1 oracle: real CuTe local_tile over an identity tensor. For every CTA, print the global
// coordinates its C / A / B views cover (colex order) and the elem_less predicate.
#include "common.hpp"
using namespace cute;

template <class Tile>
S coords(Tile const& t, int R, int C) {
  std::vector<S> v;
  for (int i = 0; i < int(size(t)); ++i) {
    auto c = t(i);
    int r = get<0>(c), k = get<1>(c);
    v.push_back("[" + jnum(r) + "," + jnum(k) + "," + jnum(elem_less(c, make_coord(R, C)) ? 1 : 0) + "]");
  }
  return jarr(v);
}

S run(int M, int N, int K) {
  auto BM = Int<32>{}; auto BN = Int<32>{}; auto BK = Int<16>{};
  auto IC = make_identity_tensor(make_shape(M, N));
  auto IA = make_identity_tensor(make_shape(M, K));
  auto IB = make_identity_tensor(make_shape(N, K));
  int gm = ceil_div(M, int(BM)), gn = ceil_div(N, int(BN)), gk = ceil_div(K, int(BK));
  std::vector<S> c, a, b;
  for (int pm = 0; pm < gm; ++pm)
    for (int pn = 0; pn < gn; ++pn)
      c.push_back(jobj({{"pid", jpair(pm, pn)},
                        {"coords", coords(local_tile(IC, make_shape(BM, BN), make_coord(pm, pn)), M, N)}}));
  for (int pm = 0; pm < gm; ++pm)
    for (int k = 0; k < gk; ++k)
      a.push_back(jobj({{"pid_m", jnum(pm)}, {"k", jnum(k)},
                        {"coords", coords(local_tile(IA, make_shape(BM, BK), make_coord(pm, k)), M, K)}}));
  for (int pn = 0; pn < gn; ++pn)
    for (int k = 0; k < gk; ++k)
      b.push_back(jobj({{"pid_n", jnum(pn)}, {"k", jnum(k)},
                        {"coords", coords(local_tile(IB, make_shape(BN, BK), make_coord(pn, k)), N, K)}}));
  // the C view of CTA (0,1) as a layout over the row-major global tensor (offsets only)
  auto gC = make_tensor(make_gmem_ptr((float*)nullptr), make_layout(make_shape(M, N), make_stride(N, Int<1>{})));
  auto v01 = local_tile(gC, make_shape(BM, BN), make_coord(0, gn > 1 ? 1 : 0));
  std::vector<S> offs;
  for (int r = 0; r < 4; ++r)
    for (int cc = 0; cc < 4; ++cc)
      offs.push_back(jnum(&v01(r, cc) - (float*)nullptr));
  return jobj({{"M", jnum(M)}, {"N", jnum(N)}, {"K", jnum(K)}, {"BM", jnum(BM)}, {"BN", jnum(BN)}, {"BK", jnum(BK)},
               {"grid", jpair(gm, gn)}, {"k_tiles", jnum(gk)}, {"c", jarr(c)}, {"a", jarr(a)}, {"b", jarr(b)},
               {"view01_offsets_4x4", jarr(offs)}});
}

int main() {
  std::printf("%s\n", jobj({{"source", jstr("cute::local_tile on make_identity_tensor")},
                            {"cases", jarr({run(64, 64, 64), run(70, 50, 40)})}}).c_str());
}
