/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

// Iris example: Classic 3-class classification on Fisher's Iris dataset
// This demonstrates multi-class classification on real botanical data
// Dataset: 150 samples, 4 features, 3 classes (perfectly balanced)

#include "iris_data.h"
#include "nm_api.h"
#include <log.h>
#include <stdio.h>
#include <stdlib.h>

static NmTensor tensor_view(const FloatMatrix *matrix) {
  return nm_tensor_2d(matrix->values, matrix->rows, matrix->cols);
}

// Normalize features to [0, 1] range
void normalize_features(FloatMatrix *X) {
  size_t rows = X->rows;
  size_t cols = X->cols;

  // Find min/max per feature
  for (size_t j = 0; j < cols; j++) {
    float min_val = sm_get(X, 0, j);
    float max_val = sm_get(X, 0, j);

    for (size_t i = 1; i < rows; i++) {
      float val = sm_get(X, i, j);
      if (val < min_val)
        min_val = val;
      if (val > max_val)
        max_val = val;
    }

    // Normalize to [0, 1]
    float range = max_val - min_val;
    if (range > 1e-6f) {
      for (size_t i = 0; i < rows; i++) {
        float val = sm_get(X, i, j);
        sm_set(X, i, j, (val - min_val) / range);
      }
    }
  }
}

// Stratified train/test split (takes every 5th sample for test)
void split_dataset_stratified(const FloatMatrix *X, const FloatMatrix *Y,
                              FloatMatrix **X_train, FloatMatrix **Y_train,
                              FloatMatrix **X_test, FloatMatrix **Y_test) {
  size_t total = X->rows;
  size_t test_size = total / 5; // 20% for test
  size_t train_size = total - test_size;

  *X_train = sm_create(train_size, X->cols);
  *Y_train = sm_create(train_size, Y->cols);
  *X_test = sm_create(test_size, X->cols);
  *Y_test = sm_create(test_size, Y->cols);

  if (!*X_train || !*Y_train || !*X_test || !*Y_test) {
    sm_destroy(*Y_train);
    sm_destroy(*X_test);
    sm_destroy(*Y_test);
    return;
  }

  size_t train_idx = 0, test_idx = 0;
  for (size_t i = 0; i < total; i++) {
    // Every 5th sample goes to test set
    if (i % 5 == 4) {
      for (size_t j = 0; j < X->cols; j++) {
        sm_set(*X_test, test_idx, j, sm_get(X, i, j));
      }
      for (size_t j = 0; j < Y->cols; j++) {
        sm_set(*Y_test, test_idx, j, sm_get(Y, i, j));
      }
      test_idx++;
    } else {
      for (size_t j = 0; j < X->cols; j++) {
        sm_set(*X_train, train_idx, j, sm_get(X, i, j));
      }
      for (size_t j = 0; j < Y->cols; j++) {
        sm_set(*Y_train, train_idx, j, sm_get(Y, i, j));
      }
      train_idx++;
    }
  }
}

int main(void) {
  log_set_level(LOG_INFO);

  const char *data_file = "examples/iris/data/iris.data";
  FloatMatrix *X_full = NULL;
  FloatMatrix *Y_full = NULL;
  size_t num_samples = 0;

  // Load dataset
  if (load_iris_dataset(data_file, &X_full, &Y_full, &num_samples) != 0) {
    log_error("Failed to load Iris dataset");
    return EXIT_FAILURE;
  }

  log_info("Dataset loaded: %zu samples, 4 features, 3 classes", num_samples);

  // Normalize features to [0, 1] for better training
  normalize_features(X_full);
  log_info("Features normalized to [0, 1] range");

  // Split into train/test (80/20) with stratification
  FloatMatrix *X_train = NULL, *Y_train = NULL;
  FloatMatrix *X_test = NULL, *Y_test = NULL;
  split_dataset_stratified(X_full, Y_full, &X_train, &Y_train, &X_test,
                           &Y_test);

  if (!X_train || !Y_train || !X_test || !Y_test) {
    log_error("Failed to split dataset");
    sm_destroy(X_full);
    sm_destroy(Y_full);
    return EXIT_FAILURE;
  }

  NmStatus s = NM_STATUS_OK;
  NmNetwork *net = NULL;
  NmSpec *spec = NULL;

  log_info("Train set: %zu samples", X_train->rows);
  log_info("Test set: %zu samples", X_test->rows);

  // Build network: 4 inputs -> 8 hidden -> 3 outputs
  spec = nm_spec_new((NmShape){.ndim = 1, .dims = {4}});
  if (!spec) {
    log_error("Failed to create model spec: %s", nm_last_error());
    goto cleanup;
  }

  s = nm_spec_dense(spec, 8, NM_ACT_RELU);
  if (s == NM_STATUS_OK) {
    s = nm_spec_dense(spec, 3, NM_ACT_SOFTMAX);
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
  fit.batch_size = 16;
  fit.learning_rate = 0.05f;
  fit.epochs = 200;

  log_info("Starting training: %zu epochs, batch_size=%zu, lr=%.4f", fit.epochs,
           fit.batch_size, fit.learning_rate);

  NmTensor x_train = tensor_view(X_train);
  NmTensor y_train = tensor_view(Y_train);
  s = nm_network_fit(net, x_train, &y_train, &fit, NULL, NULL);
  if (s != NM_STATUS_OK) {
    log_error("Training failed: %s", nm_last_error());
    goto cleanup;
  }

  NmMetrics train_metrics;
  s = nm_network_evaluate(net, tensor_view(X_train), tensor_view(Y_train),
                          NM_LOSS_CROSS_ENTROPY, &train_metrics);
  if (s != NM_STATUS_OK) {
    log_error("Training evaluation failed: %s", nm_last_error());
    goto cleanup;
  }

  log_info("Training Accuracy: %.2f%%", train_metrics.accuracy * 100.0f);

  NmMetrics test_metrics;
  s = nm_network_evaluate(net, tensor_view(X_test), tensor_view(Y_test),
                          NM_LOSS_CROSS_ENTROPY, &test_metrics);
  if (s != NM_STATUS_OK) {
    log_error("Test evaluation failed: %s", nm_last_error());
    goto cleanup;
  }

  NmTensor y_pred_test;
  s = nm_network_predict(net, tensor_view(X_test), &y_pred_test);
  if (s != NM_STATUS_OK) {
    log_error("Test prediction failed: %s", nm_last_error());
    goto cleanup;
  }

  log_info("Test Accuracy: %.2f%%", test_metrics.accuracy * 100.0f);

  // Print confusion matrix info (sample predictions)
  log_info("Sample test predictions (first 10):");
  const char *class_names[] = {"Setosa", "Versicolor", "Virginica"};
  for (size_t i = 0; i < 10 && i < X_test->rows; i++) {
    float pred_vals[3] = {y_pred_test.data[i * 3], y_pred_test.data[i * 3 + 1],
                          y_pred_test.data[i * 3 + 2]};
    float true_vals[3] = {sm_get(Y_test, i, 0), sm_get(Y_test, i, 1),
                          sm_get(Y_test, i, 2)};

    int pred_class = (pred_vals[0] > pred_vals[1])
                         ? (pred_vals[0] > pred_vals[2] ? 0 : 2)
                         : (pred_vals[1] > pred_vals[2] ? 1 : 2);
    int true_class = (true_vals[0] > 0.5f) ? 0 : (true_vals[1] > 0.5f ? 1 : 2);

    log_info("  Sample %zu: Predicted=%s (%.3f), True=%s %s", i,
             class_names[pred_class], pred_vals[pred_class],
             class_names[true_class], pred_class == true_class ? "✓" : "✗");
  }

  nm_tensor_free(&y_pred_test);

cleanup:
  sm_destroy(X_full);
  sm_destroy(Y_full);
  sm_destroy(X_train);
  sm_destroy(Y_train);
  sm_destroy(X_test);
  sm_destroy(Y_test);
  nm_spec_free(spec);
  nm_network_free(net);

  return s == NM_STATUS_OK ? EXIT_SUCCESS : EXIT_FAILURE;
}
