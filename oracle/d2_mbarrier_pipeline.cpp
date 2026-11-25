// D2 oracle: the real cutlass::PipelineState (sm90_pipeline.hpp) on the host. PipelineState is
// integer bookkeeping only; the rest of the header (mbarrier, TMA) is PTX and is never called here.
#include "common.hpp"
// host stand-ins for the two device-only names the header mentions in code we never run
static struct { unsigned x = 0, y = 0, z = 0; } threadIdx;
inline void __syncthreads() {}
#include <cutlass/pipeline/sm90_pipeline.hpp>

template <uint32_t Stages>
S walk(cutlass::PipelineState<Stages> st, int n) {
  std::vector<S> seq;
  for (int i = 0; i < n; ++i) {
    seq.push_back("[" + jnum(st.index()) + "," + jnum(st.phase()) + "," + jnum(st.count()) + "]");
    ++st;
  }
  return jarr(seq);
}

template <uint32_t Stages>
S adv(int index, int phase, int n) {
  cutlass::PipelineState<Stages> st(index, phase, 0);
  st.advance(n);
  return jobj({{"stages", jnum(Stages)}, {"start", jpair(index, phase)}, {"n", jnum(n)},
               {"got", "[" + jnum(st.index()) + "," + jnum(st.phase()) + "," + jnum(st.count()) + "]"}});
}

int main() {
  auto prod = cutlass::make_producer_start_state<cutlass::PipelineTmaAsync<4>>();
  std::vector<S> advs = {adv<4>(2, 1, 13), adv<4>(0, 0, 9), adv<3>(2, 1, 5), adv<5>(4, 1, 13),
                         adv<2>(1, 1, 1), adv<4>(3, 1, 0), adv<1>(0, 0, 3)};
  std::printf("%s\n", jobj({{"source", jstr("cutlass::PipelineState / make_producer_start_state (sm90_pipeline.hpp)")},
                            {"producer_4", walk(prod, 10)}, {"consumer_4", walk(cutlass::PipelineState<4>{}, 10)},
                            {"consumer_3", walk(cutlass::PipelineState<3>{}, 7)},
                            {"consumer_1", walk(cutlass::PipelineState<1>{}, 3)}, {"advance", jarr(advs)}}).c_str());
}
