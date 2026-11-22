/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

// Fashion-MNIST classification example
// Dataset: 60,000 training + 10,000 test grayscale images (28x28 pixels)
// Classes: 10 fashion categories (T-shirt, Trouser, Pullover, Dress, Coat,
//          Sandal, Shirt, Sneaker, Bag, Ankle boot)

#include "fashion_data.h"
#include "nm_api.h"
#include <log.h>
#include <stdio.h>
#include <stdlib.h>

// Label names for Fashion-MNIST
static const char *class_names[10] = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot"};

static NmTensor tensor_view(const FloatMatrix *matrix) {
  return nm_tensor_2d(matrix->values, matrix->rows, matrix->cols);
}

int main(void) {
  log_set_level(LOG_INFO);

  const size_t num_classes = 10;
  const size_t input_dim = 28 * 28; // 784 pixels
  const size_t hidden1_dim = 256;
  const size_t hidden2_dim = 128;
  const size_t train_size = 60000; // Full Fashion-MNIST training set
  const size_t test_size = 10000;  // Full Fashion-MNIST test set
  const size_t batch_size = 128;
  const float learning_rate = 0.01f;
  const size_t epochs = 15;

  int exit_code = EXIT_FAILURE;
  NmStatus s = NM_STATUS_OK;
  NmNetwork *net = NULL;
  NmSpec *spec = NULL;

  log_info("Loading Fashion-MNIST dataset...");
  FloatMatrix *X_train = load_fashion_images(
      "examples/fashion/data/train-images-idx3-ubyte", train_size);
  FloatMatrix *Y_train = load_fashion_labels_onehot(
      "examples/fashion/data/train-labels-idx1-ubyte", train_size, num_classes);
  FloatMatrix *X_test = load_fashion_images(
      "examples/fashion/data/t10k-images-idx3-ubyte", test_size);
  FloatMatrix *Y_test = load_fashion_labels_onehot(
      "examples/fashion/data/t10k-labels-idx1-ubyte", test_size, num_classes);

  if (!X_train || !Y_train || !X_test || !Y_test) {
    log_error("Failed to load Fashion-MNIST data");
    goto cleanup;
  }

  log_info("Dataset loaded: %zu training samples, %zu test samples", train_size,
           test_size);

  // Create a deeper network for Fashion-MNIST (more complex than MNIST)
  // Architecture: 784 -> 256 -> 128 -> 10
  log_info("Creating neural network (784 -> 256 -> 128 -> 10)...");
  spec = nm_spec_new((NmShape){.ndim = 1, .dims = {input_dim}});
  if (!spec) {
    log_error("Failed to create model spec: %s", nm_last_error());
    goto cleanup;
  }
  s = nm_spec_dense(spec, hidden1_dim, NM_ACT_RELU);
  if (s == NM_STATUS_OK) {
    s = nm_spec_dense(spec, hidden2_dim, NM_ACT_RELU);
  }
  if (s == NM_STATUS_OK) {
    s = nm_spec_dense(spec, num_classes, NM_ACT_SOFTMAX);
  }
  if (s != NM_STATUS_OK) {
    log_error("Failed to describe model: %s", nm_last_error());
    goto cleanup;
  }

  NmBuildOptions build = nm_build_options_default();
  build.seed = 42;
  net = nm_network_build(spec, &build);
  nm_spec_free(spec);
  spec = NULL;
  if (!net) {
    log_error("Failed to build model: %s", nm_last_error());
    goto cleanup;
  }

  NmFitOptions fit = nm_fit_options_default();
  fit.loss = NM_LOSS_CROSS_ENTROPY;
  fit.batch_size = batch_size;
  fit.learning_rate = learning_rate;
  fit.epochs = epochs;

  log_info("Training for %zu epochs (batch_size=%zu, lr=%.4f)...", epochs,
           batch_size, learning_rate);
  NmTensor x_train = tensor_view(X_train);
  NmTensor y_train = tensor_view(Y_train);
  s = nm_network_fit(net, x_train, &y_train, &fit, NULL, NULL);
  if (s != NM_STATUS_OK) {
    log_error("Training failed: %s", nm_last_error());
    goto cleanup;
  }

  // Evaluate on test set
  NmMetrics metrics;
  s = nm_network_evaluate(net, tensor_view(X_test), tensor_view(Y_test),
                          NM_LOSS_CROSS_ENTROPY, &metrics);
  if (s != NM_STATUS_OK) {
    log_error("Evaluation failed: %s", nm_last_error());
    goto cleanup;
  }

  log_info("Final Test Accuracy: %.2f%%", metrics.accuracy * 100.0f);

  // Show a few predictions
  log_info("\nSample Predictions (first 10 test images):");
  NmTensor y_pred;
  s = nm_network_predict(net, tensor_view(X_test), &y_pred);
  if (s != NM_STATUS_OK) {
    log_error("Prediction failed: %s", nm_last_error());
    goto cleanup;
  }
  for (size_t i = 0; i < 10 && i < test_size; ++i) {
    // Find predicted class (argmax)
    size_t pred_class = 0;
    float max_prob = y_pred.data[i * num_classes];
    for (size_t c = 1; c < num_classes; ++c) {
      if (y_pred.data[i * num_classes + c] > max_prob) {
        max_prob = y_pred.data[i * num_classes + c];
        pred_class = c;
      }
    }

    // Find true class
    size_t true_class = 0;
    for (size_t c = 0; c < num_classes; ++c) {
      if (Y_test->values[i * num_classes + c] > 0.5f) {
        true_class = c;
        break;
      }
    }

    log_info("  Sample %2zu: Predicted=%s (%.3f), True=%s %s", i,
             class_names[pred_class], max_prob, class_names[true_class],
             pred_class == true_class ? "✓" : "✗");
  }

  nm_tensor_free(&y_pred);
  exit_code = EXIT_SUCCESS;

cleanup:
  sm_destroy(X_train);
  sm_destroy(Y_train);
  sm_destroy(X_test);
  sm_destroy(Y_test);
  nm_spec_free(spec);
  nm_network_free(net);

  return exit_code;
}
