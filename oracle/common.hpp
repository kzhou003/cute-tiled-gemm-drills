// Small helpers shared by the oracle programs. Each oracle/<stem>.cpp includes real CuTe
// headers, computes the expected answers, and prints one JSON document to stdout
// (oracle/run.sh saves it as tests/golden/<stem>.json).
#pragma once
#include <cute/tensor.hpp>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using S = std::string;
inline S jnum(long long x) { return std::to_string(x); }
inline S jstr(S const& s) {
  S o = "\"";
  for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
  return o + "\"";
}
inline S jarr(std::vector<S> const& xs) {
  S o = "[";
  for (size_t i = 0; i < xs.size(); ++i) { if (i) o += ","; o += xs[i]; }
  return o + "]";
}
inline S jobj(std::vector<std::pair<S, S>> const& kv) {
  S o = "{";
  for (size_t i = 0; i < kv.size(); ++i) { if (i) o += ","; o += jstr(kv[i].first) + ":" + kv[i].second; }
  return o + "}";
}
inline S jpair(long long a, long long b) { return "[" + jnum(a) + "," + jnum(b) + "]"; }

// cute::print(x) captured into a string (e.g. "(_4,_8):(_8,_1)").
template <class T>
S to_str(T const& x) {
  char* buf = nullptr; size_t len = 0;
  FILE* f = open_memstream(&buf, &len);
  FILE* old = stdout;
  stdout = f;
  cute::print(x);
  std::fflush(f);
  stdout = old;
  std::fclose(f);
  S s(buf, len);
  std::free(buf);
  return s;
}

// All values of a layout (or any callable with a size) in index order.
template <class L>
S values(L const& layout) {
  std::vector<S> v;
  for (int i = 0; i < int(cute::size(layout)); ++i) v.push_back(jnum(layout(i)));
  return jarr(v);
}
