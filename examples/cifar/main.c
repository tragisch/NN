/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

// CIFAR-10 classification example
// Dataset: 50,000 training + 10,000 test color images (32x32x3 RGB)
// Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
//
// IMPORTANT: CIFAR-10 is designed for CNNs. Fully-connected networks
// typically achieve only 40-50% accuracy (vs 90-95% for CNNs).
// This example demonstrates the limitations of FC networks on spatial data.

#include "cifar_data.h"
#include "nm_api.h"
#include <log.h>
#include <stdio.h>
#include <stdlib.h>

static const char *class_names[10] = {"airplane", "automobile", "bird", "cat",
                                      "deer",     "dog",        "frog", "horse",
                                      "ship",     "truck"};

static NmTensor tensor_view(const FloatMatrix *matrix) {
  return nm_tensor_2d(matrix->values, matrix->rows, matrix->cols);
}

int main(void) {
  log_set_level(LOG_INFO);

  const size_t num_classes = 10;
  const size_t input_dim = 32 * 32 * 3; // 3072 pixels (RGB)
  const size_t hidden1_dim = 512;
  const size_t hidden2_dim = 256;

  // Use a subset for faster experimentation
  // Set to 0 to use full dataset (50k train, 10k test)
  const size_t max_train_per_batch = 2000; // 2k per batch = 10k total
  const size_t max_test = 2000;

  const size_t batch_size = 128;
  const float learning_rate = 0.005f; // Lower LR for stability
  const size_t epochs = 20;

  int exit_code = EXIT_FAILURE;
  NmStatus s = NM_STATUS_OK;
  NmNetwork *net = NULL;
  NmSpec *spec = NULL;

  log_info("Loading CIFAR-10 dataset (subset: %zu train/batch, %zu test)...",
           max_train_per_batch, max_test);

  FloatMatrix *X_train = NULL, *Y_train = NULL;
  FloatMatrix *X_test = NULL, *Y_test = NULL;

  // Load training data (5 batches)
  if (load_cifar10_train("examples/cifar/data", max_train_per_batch, &X_train,
                         &Y_train) != 0) {
    log_error("Failed to load CIFAR-10 training data");
    goto cleanup;
  }

  // Load test data
  if (load_cifar10_batch("examples/cifar/data/test_batch.bin", max_test,
                         &X_test, &Y_test) != 0) {
    log_error("Failed to load CIFAR-10 test data");
    goto cleanup;
  }

  log_info("Dataset loaded: %zu training samples, %zu test samples",
           X_train->rows, X_test->rows);

  // Create a 3-layer fully-connected network
  // Note: This is suboptimal for image data - CNNs work much better!
  // Architecture: 3072 -> 512 -> 256 -> 10
  log_info("Creating neural network (3072 -> 512 -> 256 -> 10)...");
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
  log_info(
      "Note: FC networks typically reach 40-50%% on CIFAR-10 (CNNs: 90-95%%)");

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

  if (metrics.accuracy > 0.40f) {
    log_info("Good! FC networks typically achieve 40-50%% on CIFAR-10");
  } else {
    log_info(
        "Consider: more epochs, larger hidden layers, or data augmentation");
  }

  // Show a few predictions
  log_info("\nSample Predictions (first 10 test images):");
  NmTensor y_pred;
  s = nm_network_predict(net, tensor_view(X_test), &y_pred);
  if (s != NM_STATUS_OK) {
    log_error("Prediction failed: %s", nm_last_error());
    goto cleanup;
  }
  for (size_t i = 0; i < 10 && i < X_test->rows; ++i) {
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

    log_info("  Sample %2zu: Predicted=%-10s (%.3f), True=%-10s %s", i,
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
