/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

// A simple example demonstrating how to train a feedforward neural network on
// the MNIST dataset

#include "mnist_data.h"
#include "nm_api.h"
#include <log.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct {
  FloatMatrix *X_train;
  FloatMatrix *Y_train;
  FloatMatrix *X_test;
  FloatMatrix *Y_test;
} MnistDataset;

static NmTensor tensor_view(const FloatMatrix *matrix) {
  return nm_tensor_2d(matrix->values, matrix->rows, matrix->cols);
}

static void mnist_dataset_destroy(MnistDataset *ds) {
  if (!ds)
    return;
  sm_destroy(ds->X_train);
  sm_destroy(ds->Y_train);
  sm_destroy(ds->X_test);
  sm_destroy(ds->Y_test);
  ds->X_train = NULL;
  ds->Y_train = NULL;
  ds->X_test = NULL;
  ds->Y_test = NULL;
}

static bool mnist_dataset_load(MnistDataset *ds, size_t train_size,
                               size_t test_size, size_t num_classes) {
  if (!ds)
    return false;

  *ds = (MnistDataset){0};
  ds->X_train =
      load_mnist_images("examples/mnist/train-images-idx3-ubyte", train_size);
  ds->Y_train = load_mnist_labels_onehot(
      "examples/mnist/train-labels-idx1-ubyte", train_size, num_classes);
  ds->X_test =
      load_mnist_images("examples/mnist/t10k-images-idx3-ubyte", test_size);
  ds->Y_test = load_mnist_labels_onehot("examples/mnist/t10k-labels-idx1-ubyte",
                                        test_size, num_classes);

  if (!ds->X_train || !ds->Y_train || !ds->X_test || !ds->Y_test) {
    mnist_dataset_destroy(ds);
    return false;
  }
  return true;
}

int main(void) {
  log_set_level(LOG_INFO);

  const size_t num_classes = 10;
  const size_t input_dim = 28 * 28;
  const size_t hidden_dim = 128;
  const size_t train_size = 60000; // Full MNIST training set
  const size_t test_size = 10000;  // Full MNIST test set
  const size_t batch_size = 128;
  const float learning_rate = 0.02f;
  const size_t epochs = 10;

  NmStatus s;
  NmNetwork *net = NULL;
  MnistDataset ds = {0};

  // 1) Load data
  if (!mnist_dataset_load(&ds, train_size, test_size, num_classes)) {
    log_error("Failed to load MNIST data");
    return EXIT_FAILURE;
  }

  // 2) Define a tiny model: 784 -> 128 -> 10
  NmSpec *spec = nm_spec_new((NmShape){.ndim = 1, .dims = {input_dim}});
  if (!spec) {
    log_error("Failed to create model spec: %s", nm_last_error());
    mnist_dataset_destroy(&ds);
    return EXIT_FAILURE;
  }

  s = nm_spec_dense(spec, hidden_dim, NM_ACT_RELU);
  if (s == NM_STATUS_OK) {
    s = nm_spec_dense(spec, num_classes, NM_ACT_SOFTMAX);
  }
  if (s != NM_STATUS_OK) {
    log_error("Failed to describe model: %s", nm_last_error());
    nm_spec_free(spec);
    mnist_dataset_destroy(&ds);
    return EXIT_FAILURE;
  }

  NmBuildOptions build = nm_build_options_default();
  build.seed = 42;
  net = nm_network_build(spec, &build);
  nm_spec_free(spec);
  if (!net) {
    log_error("Failed to build model: %s", nm_last_error());
    mnist_dataset_destroy(&ds);
    return EXIT_FAILURE;
  }

  NmFitOptions fit = nm_fit_options_default();
  fit.loss = NM_LOSS_CROSS_ENTROPY;
  fit.batch_size = batch_size;
  fit.learning_rate = learning_rate;
  fit.epochs = epochs;

  NmTensor x_train = tensor_view(ds.X_train);
  NmTensor y_train = tensor_view(ds.Y_train);
  s = nm_network_fit(net, x_train, &y_train, &fit, NULL, NULL);
  if (s != NM_STATUS_OK) {
    log_error("Training failed: %s", nm_last_error());
    nm_network_free(net);
    mnist_dataset_destroy(&ds);
    return EXIT_FAILURE;
  }

  // 4) Evaluate
  NmMetrics metrics;
  s = nm_network_evaluate(net, tensor_view(ds.X_test), tensor_view(ds.Y_test),
                          NM_LOSS_CROSS_ENTROPY, &metrics);
  if (s != NM_STATUS_OK) {
    log_error("Evaluation failed: %s", nm_last_error());
    nm_network_free(net);
    mnist_dataset_destroy(&ds);
    return EXIT_FAILURE;
  }

  log_info("Final Test Accuracy: %.2f%%", metrics.accuracy * 100.0f);

  nm_network_free(net);
  mnist_dataset_destroy(&ds);

  return EXIT_SUCCESS;
}
