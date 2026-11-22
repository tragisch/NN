extern "C" {
#include "nm.h"
#include "sm.h"
}

#include <benchmark/benchmark.h>

// Benchmark für nm_relu
static void BM_nm_relu(benchmark::State &state) {
  size_t rows_a = state.range(0);
  size_t cols_a = state.range(1);
  const int64_t elements = (int64_t)(rows_a * cols_a);

  FloatMatrix *a = sm_create_random(rows_a, cols_a);
  if (!a) {
    state.SkipWithError("sm_create_random failed");
    return;
  }

  for (auto _ : state) {
    nm_relu(a);
    benchmark::DoNotOptimize(a->values);
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * elements);
  state.SetBytesProcessed(state.iterations() * elements * (int64_t)sizeof(float));
  state.SetLabel(nm_backend());
  sm_destroy(a);
}

BENCHMARK(BM_nm_relu)->Args({64, 64});
BENCHMARK(BM_nm_relu)->Args({128, 128});
BENCHMARK(BM_nm_relu)->Args({256, 256});
BENCHMARK(BM_nm_relu)->Args({512, 512});
BENCHMARK(BM_nm_relu)->Args({1024, 1024});
BENCHMARK(BM_nm_relu)->Args({2048, 2048});
BENCHMARK(BM_nm_relu)->Args({4096, 1024});

BENCHMARK_MAIN();
