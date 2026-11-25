// B2 oracle: real max_common_vector / max_common_layout / max_alignment for (gmem view, smem layout)
// pairs, including Swizzle<B,M,S> smem layouts, and the vector width copy.hpp's
// AutoVectorizingCopyWithAssumedAlignment would pick: gcd(mcv*bits, gcd(align_a*bits, align_b*bits, 128)).
#include "common.hpp"
#include <numeric>
using namespace cute;

using G = Layout<Shape<_32, _16>, Stride<_64, _1>>;
using G60 = Layout<Shape<_32, _16>, Stride<Int<60>, _1>>;
using GC = Layout<Shape<_32, _16>, Stride<_1, _64>>;
using SP = Layout<Shape<_32, _16>, Stride<_16, _1>>;
using SPAD = Layout<Shape<_32, _16>, Stride<Int<18>, _1>>;
using SM = Layout<Shape<_32, _16>, Stride<_1, _32>>;

template <bool WithMcl = true, class A, class B>
S pair(const char* name, A a, B b) {
  S mcl = "null";
  if constexpr (WithMcl) mcl = jstr(to_str(max_common_layout(a, b)));
  int mcv = int(max_common_vector(a, b));
  long long bits = 16;
  long long align = std::gcd(std::gcd((long long)max_alignment(a) * bits, (long long)max_alignment(b) * bits), 128LL);
  long long vec = mcv <= 1 ? bits : std::gcd(mcv * bits, align);
  return jobj({{"name", jstr(name)}, {"a", jstr(to_str(a))}, {"b", jstr(to_str(b))}, {"mcv", jnum(mcv)},
               {"mcl", mcl}, {"auto_vector_bits", jnum(vec)}});
}

template <class L>
S align(const char* name, L l) {
  return jobj({{"name", jstr(name)}, {"layout", jstr(to_str(l))}, {"max_alignment", jnum(int(max_alignment(l)))}});
}

int main() {
  auto s111 = composition(Swizzle<1, 1, 1>{}, SP{});
  auto s222 = composition(Swizzle<2, 2, 2>{}, SP{});
  auto s333 = composition(Swizzle<3, 3, 3>{}, SP{});
  std::vector<S> pairs = {
      pair("grow_splain", G{}, SP{}),    pair("grow_ssw111", G{}, s111),    pair("grow_ssw222", G{}, s222),
      pair("grow_ssw333", G{}, s333),    pair("grow_spad18", G{}, SPAD{}),  pair("grow_smmajor", G{}, SM{}),
      pair("gk60_ssw333", G60{}, s333),  pair("gcol_smmajor", GC{}, SM{}),  pair<false>("ssw333_ssw333", s333, s333),
      pair<false>("ssw222_ssw333", s222, s333),
  };
  std::vector<S> aligns = {align("grow", G{}),     align("gk60", G60{}),  align("gcol", GC{}),
                           align("srow", SP{}),    align("ssw111", s111), align("ssw222", s222),
                           align("ssw333", s333),  align("spad18", SPAD{}), align("smmajor", SM{})};
  std::printf("%s\n", jobj({{"source", jstr("cute::max_common_vector / max_common_layout / max_alignment")},
                            {"pairs", jarr(pairs)}, {"alignment", jarr(aligns)}}).c_str());
}
