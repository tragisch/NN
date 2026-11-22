extern "C" {
#include "st.h"
#include "st_conv.h"
}

#include <benchmark/benchmark.h>

#include <cstdint>
#include <string>

namespace {

struct ConvCase {
  size_t n;
  size_t c_in;
  size_t c_out;
  size_t h;
  size_t w;
  size_t k;
  size_t stride;
  size_t pad;
  StConvBackend backend;
};

static FloatTensor *make4d(size_t n, size_t c, size_t h, size_t w) {
  const size_t shape[4] = {n, c, h, w};
  return st_create(4, shape);
}

static void fill_tensor(FloatTensor *tensor, uint32_t seed) {
  float *data = st_tensor_mutable_data(tensor);
  const size_t numel = st_tensor_numel(tensor);
  if (!data) {
    return;
  }
  for (size_t i = 0; i < numel; ++i) {
    seed = seed * 1664525u + 1013904223u;
    data[i] = static_cast<float>((seed >> 8) & 0xFFFFu) / 65535.0f - 0.5f;
  }
}

static const char *backend_name(StConvBackend backend) {
  switch (backend) {
  case ST_CONV_BACKEND_AUTO:
    return "AUTO";
  case ST_CONV_BACKEND_REFERENCE:
    return "REF";
  case ST_CONV_BACKEND_CPU_OPT:
    return "CPU_OPT";
  case ST_CONV_BACKEND_GEMM:
    return "GEMM";
  case ST_CONV_BACKEND_MPS:
    return "MPS";
  case ST_CONV_BACKEND_BNNS:
    return "BNNS";
  }
  return "UNKNOWN";
}

static void BM_st_conv2d_canonical(benchmark::State &state) {
  const ConvCase cfg = {
      static_cast<size_t>(state.range(0)),
      static_cast<size_t>(state.range(1)),
      static_cast<size_t>(state.range(2)),
      static_cast<size_t>(state.range(3)),
      static_cast<size_t>(state.range(4)),
      static_cast<size_t>(state.range(5)),
      static_cast<size_t>(state.range(6)),
      static_cast<size_t>(state.range(7)),
      static_cast<StConvBackend>(state.range(8)),
  };

  StConv2dParams params = st_conv2d_default_params();
  params.stride_h = cfg.stride;
  params.stride_w = cfg.stride;
  params.pad_h = cfg.pad;
  params.pad_w = cfg.pad;
  params.backend = cfg.backend;

  size_t out_h = 0;
  size_t out_w = 0;
  if (!st_conv2d_output_hw(cfg.h, cfg.w, cfg.k, cfg.k, &params, &out_h,
                           &out_w)) {
    state.SkipWithError("st_conv2d_output_hw failed");
    return;
  }

  FloatTensor *input = make4d(cfg.n, cfg.c_in, cfg.h, cfg.w);
  FloatTensor *weight = make4d(cfg.c_out, cfg.c_in, cfg.k, cfg.k);
  FloatTensor *output = make4d(cfg.n, cfg.c_out, out_h, out_w);
  if (!input || !weight || !output) {
    state.SkipWithError("Failed to allocate tensors");
    st_destroy(input);
    st_destroy(weight);
    st_destroy(output);
    return;
  }

  fill_tensor(input, 1u);
  fill_tensor(weight, 2u);

  for (auto _ : state) {
    if (!st_conv2d_nchw(input, weight, nullptr, &params, output)) {
      state.SkipWithError("st_conv2d_nchw failed");
      break;
    }
    st_tensor_sync(output);
    benchmark::DoNotOptimize(st_tensor_data(output));
    benchmark::ClobberMemory();
  }

  const int64_t out_elems = static_cast<int64_t>(cfg.n) *
                            static_cast<int64_t>(cfg.c_out) *
                            static_cast<int64_t>(out_h) *
                            static_cast<int64_t>(out_w);
  state.SetItemsProcessed(state.iterations() * out_elems);
  state.SetLabel(std::string("conv2d/") + backend_name(cfg.backend) +
                 " observed=" + st_conv2d_last_backend());

  st_destroy(output);
  st_destroy(weight);
  st_destroy(input);
}

BENCHMARK(BM_st_conv2d_canonical)
    ->Args({1, 8, 8, 16, 16, 3, 1, 1, ST_CONV_BACKEND_AUTO})
    ->Args({1, 8, 8, 16, 16, 3, 1, 1, ST_CONV_BACKEND_GEMM})
    ->Args({4, 32, 64, 56, 56, 3, 1, 1, ST_CONV_BACKEND_AUTO})
    ->Args({4, 32, 64, 56, 56, 3, 1, 1, ST_CONV_BACKEND_GEMM})
    ->Args({8, 64, 128, 112, 112, 3, 1, 1, ST_CONV_BACKEND_AUTO})
    ->Args({8, 64, 128, 112, 112, 3, 1, 1, ST_CONV_BACKEND_MPS})
    ->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN();
