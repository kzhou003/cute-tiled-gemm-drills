// F1 oracle: the local_partition calls of solutions/f1_simt_cute_gemm.cu, run on the host over
// identity tensors, so every element's value is its (m, k) / (m, n) coordinate inside the CTA tile.
//   copy:    local_partition(tile(128,8), ThrA=(32,8):(8,1), t)
//   compute: local_partition(sA(128,8), ThrC=(16,16):(1,16), t, Step<_1,X>)  rows of sA read by t
//            local_partition(sB(128,8), ThrC, t, Step<X,_1>)                  rows of sB read by t
//            local_partition(gC(128,128), ThrC, t, Step<_1,_1>)               the 64 outputs of t
#include "common.hpp"
using namespace cute;

using ThrA = Layout<Shape<_32, _8>, Stride<_8, _1>>;
using ThrC = Layout<Shape<_16, _16>, Stride<_1, _16>>;

template <class T>
S coords(T const& t) {
  std::vector<S> v;
  for (int i = 0; i < int(size(t)); ++i) v.push_back(jpair(get<0>(t(i)), get<1>(t(i))));
  return jarr(v);
}

int main() {
  auto cA = make_identity_tensor(make_shape(_128{}, _8{}));
  auto cC = make_identity_tensor(make_shape(_128{}, _128{}));
  std::vector<std::pair<S, S>> threads;
  for (int t : {0, 1, 37, 100, 255}) {
    auto tA = local_partition(cA, ThrA{}, t);
    auto tCsA = local_partition(cA, ThrC{}, t, Step<_1, X>{});
    auto tCsB = local_partition(cA, ThrC{}, t, Step<X, _1>{});
    auto tCgC = local_partition(cC, ThrC{}, t, Step<_1, _1>{});
    std::vector<S> arow, brow;
    for (int a = 0; a < int(size<0>(tCsA)); ++a) arow.push_back(jnum(get<0>(tCsA(a, 0))));
    for (int b = 0; b < int(size<0>(tCsB)); ++b) brow.push_back(jnum(get<0>(tCsB(b, 0))));
    threads.push_back({jnum(t), jobj({{"copy", coords(tA)}, {"copy_shape", jstr(to_str(shape(tA)))},
                                      {"c", coords(tCgC)}, {"c_shape", jstr(to_str(shape(tCgC)))},
                                      {"a_rows", jarr(arow)}, {"b_rows", jarr(brow)},
                                      {"a_shape", jstr(to_str(shape(tCsA)))}})});
  }
  // the CTA views for a 256x384x64 problem: shapes printed by the kernel's three local_tile calls
  auto mA = make_identity_tensor(make_shape(256, 64));
  auto gA = local_tile(mA, make_shape(_128{}, _128{}, _8{}), make_coord(1, 2, _), Step<_1, X, _1>{});
  std::printf("%s\n", jobj({{"source", jstr("local_partition over identity tensors, F1 layouts")},
                            {"threads", jobj(threads)},
                            {"gA_shape", jstr(to_str(shape(gA)))},
                            {"gA_first", jpair(get<0>(gA(0, 0, 0)), get<1>(gA(0, 0, 0)))},
                            {"gA_k3_first", jpair(get<0>(gA(0, 0, 3)), get<1>(gA(0, 0, 3)))}}).c_str());
}
