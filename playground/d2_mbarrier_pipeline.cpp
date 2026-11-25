// D2: the real cutlass::PipelineState on the host (integer bookkeeping; mbarrier itself is PTX).
#include <cstdio>
static struct { unsigned x = 0, y = 0, z = 0; } threadIdx;
inline void __syncthreads() {}
#include <cutlass/pipeline/sm90_pipeline.hpp>

int main() {
  auto prod = cutlass::make_producer_start_state<cutlass::PipelineTmaAsync<4>>();
  cutlass::PipelineState<4> cons{};
  printf("  k | producer (index,phase) | consumer (index,phase)\n");
  for (int k = 0; k < 9; ++k) {
    printf(" %2d |        (%d,%d)           |        (%d,%d)\n", k, prod.index(), prod.phase(), cons.index(), cons.phase());
    ++prod;
    ++cons;
  }
  cutlass::PipelineState<4> s(2, 1, 0);
  s.advance(13);
  printf("PipelineState<4>(2,1,0).advance(13) = (%d,%d,%d)\n", s.index(), s.phase(), s.count());
}
