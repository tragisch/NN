extern "C" {
#include "nm.h"
}

#include <benchmark/benchmark.h>

#include <cstdint>
#include <string>

namespace {

static FloatMatrix *make_input(size_t n, size_t c, size_t h, size_t w) {
  FloatMatrix *x = sm_create(n, c * h * w);
  if (!x) {
    return nullptr;
  }
  const size_t total = x->rows * x->cols;
  for (size_t i = 0; i < total; ++i) {
    x->values[i] = static_cast<float>((i % 113u) * 0.01f);
  }
  return x;
}

static FloatMatrix *make_onehot_targets(size_t n, size_t classes) {
  FloatMatrix *y = sm_create(n, classes);
  if (!y) {
    return nullptr;
  }
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < classes; ++c) {
      y->values[r * classes + c] = 0.0f;
    }
    y->values[r * classes + (r % classes)] = 1.0f;
  }
  return y;
}

static NmModel *create_conv_model(size_t c, size_t h, size_t w, size_t filters,
                                  size_t kernel, NmConvAlgo algo,
                                  NmTensorLayout layout) {
  NmShape input_shape;
  input_shape.ndim = 3;
  input_shape.dims[0] = c;
  input_shape.dims[1] = h;
  input_shape.dims[2] = w;

  NmModel *m = nm_create(input_shape);
  if (!m) {
    return nullptr;
  }

  if (nm_set_tensor_layout(m, layout) != NM_STATUS_OK) {
    nm_free(m);
    return nullptr;
  }

  NmConv2dConfig conv_cfg;
  conv_cfg.filters = filters;
  conv_cfg.kernel_size = kernel;
  conv_cfg.stride = 1;
  conv_cfg.padding = kernel / 2;
  conv_cfg.algo = algo;
  conv_cfg.layout = layout;
  conv_cfg.activation = NM_ACT_RELU;

  if (nm_add_conv2d(m, &conv_cfg) != NM_STATUS_OK ||
      nm_add_flatten(m) != NM_STATUS_OK) {
    nm_free(m);
    return nullptr;
  }

  NmDenseConfig out_cfg;
  out_cfg.units = 10;
  out_cfg.activation = NM_ACT_SOFTMAX;
  if (nm_add_dense(m, &out_cfg) != NM_STATUS_OK) {
    nm_free(m);
    return nullptr;
  }

  NmCompileConfig cc = nm_compile_defaults();
  cc.loss_type = NM_LOSS_CROSS_ENTROPY;
  if (nm_compile(m, &cc) != NM_STATUS_OK) {
    nm_free(m);
    return nullptr;
  }

  return m;
}

static NmModel *create_dense_model(size_t in_features, size_t hidden,
                                   size_t out_features) {
  NmShape input_shape;
  input_shape.ndim = 1;
  input_shape.dims[0] = in_features;
  input_shape.dims[1] = 0;
  input_shape.dims[2] = 0;

  NmModel *m = nm_create(input_shape);
  if (!m) {
    return nullptr;
  }

  NmDenseConfig d1;
  d1.units = hidden;
  d1.activation = NM_ACT_RELU;

  NmDenseConfig d2;
  d2.units = out_features;
  d2.activation = NM_ACT_SOFTMAX;

  if (nm_add_dense(m, &d1) != NM_STATUS_OK ||
      nm_add_dense(m, &d2) != NM_STATUS_OK) {
    nm_free(m);
    return nullptr;
  }

  NmCompileConfig cc = nm_compile_defaults();
  cc.loss_type = NM_LOSS_CROSS_ENTROPY;
  if (nm_compile(m, &cc) != NM_STATUS_OK) {
    nm_free(m);
    return nullptr;
  }

  return m;
}

static const char *algo_name(NmConvAlgo a) {
  switch (a) {
  case NM_CONV_ALGO_AUTO:
    return "AUTO";
  case NM_CONV_ALGO_DIRECT:
    return "DIRECT";
  case NM_CONV_ALGO_GEMM:
    return "GEMM";
  case NM_CONV_ALGO_REFERENCE:
    return "REF";
  }
  return "UNKNOWN";
}

static const char *layout_name(NmTensorLayout l) {
  switch (l) {
  case NM_LAYOUT_NCHW:
    return "NCHW";
  case NM_LAYOUT_NHWC:
    return "NHWC";
  }
  return "UNKNOWN";
}

static const char *parallel_name(int mode) {
  switch (mode) {
  case 0:
    return "serial";
  case 1:
    return "parallel_det";
  case 2:
    return "parallel_nondet";
  default:
    return "unknown";
  }
}

static void BM_nm_seq_conv_predict(benchmark::State &state) {
  const size_t samples = static_cast<size_t>(state.range(0));
  const size_t c = static_cast<size_t>(state.range(1));
  const size_t h = static_cast<size_t>(state.range(2));
  const size_t w = static_cast<size_t>(state.range(3));
  const size_t filters = static_cast<size_t>(state.range(4));
  const size_t kernel = static_cast<size_t>(state.range(5));
  const NmConvAlgo algo = static_cast<NmConvAlgo>(state.range(6));
  const NmTensorLayout layout = static_cast<NmTensorLayout>(state.range(7));

  FloatMatrix *x = make_input(samples, c, h, w);
  NmModel *m = create_conv_model(c, h, w, filters, kernel, algo, layout);
  if (!x || !m) {
    state.SkipWithError("setup failed");
    sm_destroy(x);
    nm_free(m);
    return;
  }

  for (auto _ : state) {
    FloatMatrix *pred = nullptr;
    NmStatus s = nm_predict(m, x, &pred);
    if (s != NM_STATUS_OK || !pred) {
      state.SkipWithError("nm_predict failed");
      sm_destroy(pred);
      break;
    }
    benchmark::DoNotOptimize(pred->values);
    benchmark::ClobberMemory();
    sm_destroy(pred);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(samples));
  std::string label = std::string(algo_name(algo)) + "/" + layout_name(layout) +
                      " backend=" + nm_model_backend(m);
  state.SetLabel(label);

  sm_destroy(x);
  nm_free(m);
}

BENCHMARK(BM_nm_seq_conv_predict)
    ->Args({8, 3, 32, 32, 16, 3, NM_CONV_ALGO_AUTO, NM_LAYOUT_NCHW})
    ->Args({8, 3, 32, 32, 16, 3, NM_CONV_ALGO_DIRECT, NM_LAYOUT_NCHW})
    ->Args({8, 3, 32, 32, 16, 3, NM_CONV_ALGO_GEMM, NM_LAYOUT_NCHW})
    ->Args({8, 3, 32, 32, 16, 3, NM_CONV_ALGO_AUTO, NM_LAYOUT_NHWC})
    ->Args({8, 3, 32, 32, 32, 5, NM_CONV_ALGO_AUTO, NM_LAYOUT_NCHW})
    ->Args({8, 3, 32, 32, 32, 5, NM_CONV_ALGO_DIRECT, NM_LAYOUT_NCHW})
    ->Args({8, 3, 32, 32, 32, 5, NM_CONV_ALGO_GEMM, NM_LAYOUT_NCHW})
    ->Unit(benchmark::kMillisecond);

static void BM_nm_seq_dense_train_parallel(benchmark::State &state) {
  const size_t samples = static_cast<size_t>(state.range(0));
  const size_t in_features = static_cast<size_t>(state.range(1));
  const size_t hidden = static_cast<size_t>(state.range(2));
  const size_t classes = static_cast<size_t>(state.range(3));
  const int parallel_mode = static_cast<int>(state.range(4));
  const size_t workers = static_cast<size_t>(state.range(5));

  FloatMatrix *x = sm_create(samples, in_features);
  FloatMatrix *y = make_onehot_targets(samples, classes);
  NmModel *m = create_dense_model(in_features, hidden, classes);
  if (!x || !y || !m) {
    state.SkipWithError("setup failed");
    sm_destroy(x);
    sm_destroy(y);
    nm_free(m);
    return;
  }

  for (size_t i = 0; i < x->rows * x->cols; ++i) {
    x->values[i] = static_cast<float>((i % 97u) * 0.005f);
  }

  NmBatchParallelConfig pcfg;
  pcfg.enabled = (parallel_mode != 0);
  pcfg.deterministic = (parallel_mode != 2);
  pcfg.num_workers = workers;
  if (nm_set_batch_parallel(m, &pcfg) != NM_STATUS_OK) {
    state.SkipWithError("nm_set_batch_parallel failed");
    sm_destroy(x);
    sm_destroy(y);
    nm_free(m);
    return;
  }

  NmTrainConfig tc = nm_train_defaults();
  tc.epochs = 1;
  tc.batch_size = 32;
  tc.learning_rate = 0.01f;
  tc.verbose = false;

  for (auto _ : state) {
    NmStatus s = nm_fit(m, x, y, &tc);
    if (s != NM_STATUS_OK) {
      state.SkipWithError("nm_fit failed");
      break;
    }
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(samples));
  std::string label = std::string(parallel_name(parallel_mode)) +
                      " workers=" + std::to_string(workers);
  state.SetLabel(label);

  sm_destroy(x);
  sm_destroy(y);
  nm_free(m);
}

BENCHMARK(BM_nm_seq_dense_train_parallel)
    ->Args({512, 128, 64, 10, 0, 0})
    ->Args({512, 128, 64, 10, 1, 0})
    ->Args({512, 128, 64, 10, 1, 2})
    ->Args({512, 128, 64, 10, 1, 4})
    ->Args({512, 128, 64, 10, 2, 0})
    ->Args({512, 128, 64, 10, 2, 2})
    ->Args({512, 128, 64, 10, 2, 4})
    ->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN();
