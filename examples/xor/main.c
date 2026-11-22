/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

// XOR example: A simple non-linearly separable binary classification task
// This demonstrates that the network can learn non-linear decision boundaries

#include "nm_api.h"
#include <log.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  log_set_level(LOG_INFO);

  // XOR dataset: 4 samples, 2 inputs, 2 outputs (one-hot encoded)
  // (0,0) -> (1,0) [class 0]
  // (0,1) -> (0,1) [class 1]
  // (1,0) -> (0,1) [class 1]
  // (1,1) -> (1,0) [class 0]

  float X_data[8] = {
      0.0f, 0.0f, // XOR(0,0) = 0
      0.0f, 1.0f, // XOR(0,1) = 1
      1.0f, 0.0f, // XOR(1,0) = 1
      1.0f, 1.0f  // XOR(1,1) = 0
  };

  float Y_data[8] = {
      1.0f, 0.0f, // Class 0 (one-hot)
      0.0f, 1.0f, // Class 1
      0.0f, 1.0f, // Class 1
      1.0f, 0.0f  // Class 0
  };

  // Build a network: 2 inputs -> 8 hidden neurons -> 2 outputs
  // We need hidden layer to solve non-linear problem
  NmSpec *spec = nm_spec_new((NmShape){.ndim = 1, .dims = {2}});
  if (!spec) {
    log_error("Failed to create model spec: %s", nm_last_error());
    return EXIT_FAILURE;
  }

  NmStatus s = nm_spec_dense(spec, 8, NM_ACT_RELU);
  if (s == NM_STATUS_OK) {
    s = nm_spec_dense(spec, 2, NM_ACT_SOFTMAX);
  }
  if (s != NM_STATUS_OK) {
    log_error("Failed to describe model: %s", nm_last_error());
    nm_spec_free(spec);
    return EXIT_FAILURE;
  }

  NmBuildOptions build = nm_build_options_default();
  build.seed = 42;
  NmNetwork *net = nm_network_build(spec, &build);
  nm_spec_free(spec);
  if (!net) {
    log_error("Failed to build model: %s", nm_last_error());
    return EXIT_FAILURE;
  }

  NmTensor inputs = nm_tensor_2d(X_data, 4, 2);
  NmTensor targets = nm_tensor_2d(Y_data, 4, 2);
  NmFitOptions fit = nm_fit_options_default();
  fit.loss = NM_LOSS_CROSS_ENTROPY;
  fit.batch_size = 4; // Full batch for small dataset
  fit.learning_rate = 0.5f;
  fit.epochs = 5000;

  log_info("Starting training for %zu epochs", fit.epochs);
  s = nm_network_fit(net, inputs, &targets, &fit, NULL, NULL);
  if (s != NM_STATUS_OK) {
    log_error("Training failed: %s", nm_last_error());
    nm_network_free(net);
    return EXIT_FAILURE;
  }

  // Evaluate on the same data (since we want to memorize it)
  NmTensor y_pred;
  s = nm_network_predict(net, inputs, &y_pred);
  if (s != NM_STATUS_OK) {
    log_error("Prediction failed: %s", nm_last_error());
    nm_network_free(net);
    return EXIT_FAILURE;
  }

  NmMetrics metrics;
  s = nm_network_evaluate(net, inputs, targets, NM_LOSS_CROSS_ENTROPY,
                          &metrics);
  if (s != NM_STATUS_OK) {
    log_error("Evaluation failed: %s", nm_last_error());
    nm_tensor_free(&y_pred);
    nm_network_free(net);
    return EXIT_FAILURE;
  }

  log_info("Final Training Accuracy: %.2f%%", metrics.accuracy * 100.0f);

  // Print detailed predictions
  log_info("Detailed predictions:");
  for (size_t i = 0; i < 4; i++) {
    float input_0 = X_data[i * 2];
    float input_1 = X_data[i * 2 + 1];
    float pred_0 = y_pred.data[i * 2];
    float pred_1 = y_pred.data[i * 2 + 1];
    float true_0 = Y_data[i * 2];
    float true_1 = Y_data[i * 2 + 1];
    int predicted_class = pred_0 > pred_1 ? 0 : 1;
    int true_class = true_0 > true_1 ? 0 : 1;

    log_info("  XOR(%.0f, %.0f) -> Predicted: class %d (%.3f, %.3f), "
             "True: class %d (%.0f, %.0f) %s",
             input_0, input_1, predicted_class, pred_0, pred_1, true_class,
             true_0, true_1, predicted_class == true_class ? "✓" : "✗");
  }

  nm_tensor_free(&y_pred);
  nm_network_free(net);

  return EXIT_SUCCESS;
}
