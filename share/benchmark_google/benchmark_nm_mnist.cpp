extern "C" {
#include "mnist_data.h"
#include "nm.h"
}

#include <benchmark/benchmark.h>

static NmModel *
create_mnist_net(size_t input_dim, size_t hidden_dim, size_t num_classes,
                 const NmOptimizerConfig *optimizer_cfg = nullptr) {
  NmShape input_shape = {};
  input_shape.ndim = 1;
  input_shape.dims[0] = input_dim;

  NmModel *net = nm_create(input_shape);
  if (!net) {
    return nullptr;
  }

  NmDenseConfig hidden_cfg = {};
  hidden_cfg.units = hidden_dim;
  hidden_cfg.activation = NM_ACT_RELU;
  if (nm_add_dense(net, &hidden_cfg) != NM_STATUS_OK) {
    nm_free(net);
    return nullptr;
  }

  NmDenseConfig output_cfg = {};
  output_cfg.units = num_classes;
  output_cfg.activation = NM_ACT_SOFTMAX;
  if (nm_add_dense(net, &output_cfg) != NM_STATUS_OK) {
    nm_free(net);
    return nullptr;
  }

  NmCompileConfig compile_cfg = nm_compile_defaults();
  compile_cfg.loss_type = NM_LOSS_CROSS_ENTROPY;
  if (nm_compile(net, &compile_cfg) != NM_STATUS_OK) {
    nm_free(net);
    return nullptr;
  }

  if (optimizer_cfg && nm_set_optimizer(net, optimizer_cfg) != NM_STATUS_OK) {
    nm_free(net);
    return nullptr;
  }

  return net;
}

static void BM_nm_mnist_train_one_epoch(benchmark::State &state) {
  const size_t samples = (size_t)state.range(0);
  const size_t hidden = (size_t)state.range(1);
  const size_t num_classes = 10;
  const size_t input_dim = 28 * 28;

  nm_set_seed(42);
  FloatMatrix *X =
      load_mnist_images("examples/mnist/train-images-idx3-ubyte", samples);
  FloatMatrix *Y = load_mnist_labels_onehot(
      "examples/mnist/train-labels-idx1-ubyte", samples, num_classes);

  if (!X || !Y) {
    state.SkipWithError("Failed to load MNIST training data");
    sm_destroy(X);
    sm_destroy(Y);
    return;
  }

  NmModel *net = create_mnist_net(input_dim, hidden, num_classes);
  if (!net) {
    state.SkipWithError("Failed to initialize network");
    sm_destroy(X);
    sm_destroy(Y);
    return;
  }

  NmTrainConfig train_cfg = nm_train_defaults();
  train_cfg.batch_size = 128;
  train_cfg.learning_rate = 0.02f;
  train_cfg.epochs = 1;
  train_cfg.verbose = false;

  for (auto _ : state) {
    if (nm_fit_epoch(net, X, Y, &train_cfg) != NM_STATUS_OK) {
      state.SkipWithError("train epoch failed");
      break;
    }
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * (int64_t)samples);
  state.SetLabel(nm_backend());

  nm_free(net);
  sm_destroy(X);
  sm_destroy(Y);
}

static void BM_nm_mnist_train_one_epoch_sgd(benchmark::State &state) {
  const size_t samples = (size_t)state.range(0);
  const size_t hidden = (size_t)state.range(1);
  const size_t num_classes = 10;
  const size_t input_dim = 28 * 28;

  nm_set_seed(42);
  FloatMatrix *X =
      load_mnist_images("examples/mnist/train-images-idx3-ubyte", samples);
  FloatMatrix *Y = load_mnist_labels_onehot(
      "examples/mnist/train-labels-idx1-ubyte", samples, num_classes);

  if (!X || !Y) {
    state.SkipWithError("Failed to load MNIST training data");
    sm_destroy(X);
    sm_destroy(Y);
    return;
  }

  NmOptimizerConfig opt_cfg = nm_optimizer_defaults();
  opt_cfg.type = NM_OPT_SGD;
  opt_cfg.learning_rate = 0.02f;

  NmModel *net = create_mnist_net(input_dim, hidden, num_classes, &opt_cfg);
  if (!net) {
    state.SkipWithError("Failed to initialize network");
    sm_destroy(X);
    sm_destroy(Y);
    return;
  }

  NmTrainConfig train_cfg = nm_train_defaults();
  train_cfg.batch_size = 128;
  train_cfg.learning_rate = 0.02f;
  train_cfg.epochs = 1;
  train_cfg.verbose = false;

  for (auto _ : state) {
    if (nm_fit_epoch(net, X, Y, &train_cfg) != NM_STATUS_OK) {
      state.SkipWithError("train epoch failed");
      break;
    }
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * (int64_t)samples);
  state.SetLabel(nm_backend());

  nm_free(net);
  sm_destroy(X);
  sm_destroy(Y);
}

static void BM_nm_mnist_predict(benchmark::State &state) {
  const size_t samples = (size_t)state.range(0);
  const size_t hidden = (size_t)state.range(1);
  const size_t num_classes = 10;
  const size_t input_dim = 28 * 28;

  nm_set_seed(42);
  FloatMatrix *X =
      load_mnist_images("examples/mnist/t10k-images-idx3-ubyte", samples);

  if (!X) {
    state.SkipWithError("Failed to load MNIST test images");
    return;
  }

  NmModel *net = create_mnist_net(input_dim, hidden, num_classes);
  if (!net) {
    state.SkipWithError("Failed to initialize network");
    sm_destroy(X);
    return;
  }

  for (auto _ : state) {
    FloatMatrix *pred = nullptr;
    if (nm_predict(net, X, &pred) != NM_STATUS_OK) {
      state.SkipWithError("predict failed");
      break;
    }
    benchmark::DoNotOptimize(pred->values);
    sm_destroy(pred);
  }

  state.SetItemsProcessed(state.iterations() * (int64_t)samples);
  state.SetLabel(nm_backend());

  nm_free(net);
  sm_destroy(X);
}

static void BM_nm_mnist_predict_into(benchmark::State &state) {
  const size_t samples = (size_t)state.range(0);
  const size_t hidden = (size_t)state.range(1);
  const size_t num_classes = 10;
  const size_t input_dim = 28 * 28;

  nm_set_seed(42);
  FloatMatrix *X =
      load_mnist_images("examples/mnist/t10k-images-idx3-ubyte", samples);

  if (!X) {
    state.SkipWithError("Failed to load MNIST test images");
    return;
  }

  NmModel *net = create_mnist_net(input_dim, hidden, num_classes);
  if (!net) {
    state.SkipWithError("Failed to initialize network");
    sm_destroy(X);
    return;
  }

  FloatMatrix *pred = sm_create(samples, num_classes);
  if (!pred) {
    state.SkipWithError("Failed to allocate prediction buffer");
    nm_free(net);
    sm_destroy(X);
    return;
  }

  for (auto _ : state) {
    if (nm_predict_into(net, X, pred) != NM_STATUS_OK) {
      state.SkipWithError("predict_into failed");
      break;
    }
    benchmark::DoNotOptimize(pred->values);
  }

  state.SetItemsProcessed(state.iterations() * (int64_t)samples);
  state.SetLabel(nm_backend());

  sm_destroy(pred);
  nm_free(net);
  sm_destroy(X);
}

BENCHMARK(BM_nm_mnist_train_one_epoch)->Args({1000, 128})->Args({5000, 128});
BENCHMARK(BM_nm_mnist_train_one_epoch_sgd)->Args({1000, 128});
BENCHMARK(BM_nm_mnist_predict)
    ->Args({256, 128})
    ->Args({1024, 128})
    ->Args({2048, 128});
BENCHMARK(BM_nm_mnist_predict_into)->Args({256, 128})->Args({1024, 128});

BENCHMARK_MAIN();
