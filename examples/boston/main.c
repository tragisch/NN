/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

// Boston Housing example: Regression on real estate price prediction
// This demonstrates continuous value prediction using MSE loss
// Dataset: 506 samples, 13 features, 1 continuous target (median home value in
// $1000s)

#include "nm_api.h"
#include <log.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Synthetic Boston Housing data (simplified for demo)
// Real dataset would be loaded from CSV
// Features: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B,
// LSTAT Target: MEDV (median value in $1000s)

#define NUM_SAMPLES 100
#define NUM_FEATURES 13

int main(void) {
  log_set_level(LOG_INFO);
  srand(42);

  // Generate synthetic housing data
  // y = 5.0 + 2.0*RM + 3.0*LSTAT + ... (simplified model)
  float X_data[NUM_SAMPLES * NUM_FEATURES];
  float Y_data[NUM_SAMPLES];

  for (size_t i = 0; i < NUM_SAMPLES; i++) {
    // Generate 13 random features [0, 1]
    for (size_t j = 0; j < NUM_FEATURES; j++) {
      X_data[i * NUM_FEATURES + j] = (float)(rand()) / RAND_MAX;
    }

    // Synthetic target: weighted combination of features + noise
    float target = 10.0f;                           // Base price ~$10k
    target += 5.0f * X_data[i * NUM_FEATURES + 5];  // RM (rooms)
    target += 3.0f * X_data[i * NUM_FEATURES + 12]; // LSTAT (lower status %)
    target += 2.0f * X_data[i * NUM_FEATURES + 0];  // CRIM (crime rate)
    target -= 1.0f * X_data[i * NUM_FEATURES + 8];  // RAD (radial highways)
    target += 0.5f * ((float)(rand()) / RAND_MAX - 0.5f); // Add noise

    // Normalize to [0, 1] range for Sigmoid output
    target = target / 55.0f;                   // Divide by max expected price
    target = fmaxf(0.0f, fminf(1.0f, target)); // Clamp to [0, 1]

    Y_data[i] = target;
  }

  log_info("Dataset created: %zu samples, %zu features", NUM_SAMPLES,
           NUM_FEATURES);

  // Build regression network: 13 inputs -> 32 hidden -> 8 hidden -> 1 output
  NmSpec *spec = nm_spec_new((NmShape){.ndim = 1, .dims = {NUM_FEATURES}});
  if (!spec) {
    log_error("Failed to create model spec: %s", nm_last_error());
    return EXIT_FAILURE;
  }
  NmStatus s = NM_STATUS_OK;
  s = nm_spec_dense(spec, 32, NM_ACT_RELU);
  if (s == NM_STATUS_OK) {
    s = nm_spec_dense(spec, 8, NM_ACT_RELU);
  }
  if (s == NM_STATUS_OK) {
    s = nm_spec_dense(spec, 1, NM_ACT_SIGMOID);
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

  NmTensor inputs = nm_tensor_2d(X_data, NUM_SAMPLES, NUM_FEATURES);
  NmTensor targets = nm_tensor_2d(Y_data, NUM_SAMPLES, 1);
  NmFitOptions fit = nm_fit_options_default();
  fit.loss = NM_LOSS_MSE;
  fit.batch_size = 32;
  fit.learning_rate = 0.01f;
  fit.epochs = 200;

  log_info("Starting training: %zu epochs, batch_size=%zu, lr=%.4f", fit.epochs,
           fit.batch_size, fit.learning_rate);

  s = nm_network_fit(net, inputs, &targets, &fit, NULL, NULL);
  if (s != NM_STATUS_OK) {
    log_error("Training failed: %s", nm_last_error());
    nm_network_free(net);
    return EXIT_FAILURE;
  }

  // Evaluate on training data
  NmMetrics metrics;
  s = nm_network_evaluate(net, inputs, targets, NM_LOSS_MSE, &metrics);
  if (s != NM_STATUS_OK) {
    log_error("Evaluation failed: %s", nm_last_error());
    nm_network_free(net);
    return EXIT_FAILURE;
  }

  // Compute MSE (Mean Squared Error)
  float mse = metrics.loss;
  float rmse = sqrtf(mse);

  log_info("Final Training MSE: %.6f (normalized [0,1])", mse);
  log_info("Final Training RMSE: %.4f (normalized units)", rmse);

  // Compute MAE (Mean Absolute Error) for interpretability
  NmTensor y_pred;
  s = nm_network_predict(net, inputs, &y_pred);
  if (s != NM_STATUS_OK) {
    log_error("Prediction failed: %s", nm_last_error());
    nm_network_free(net);
    return EXIT_FAILURE;
  }
  float total_mae = 0.0f;
  for (size_t i = 0; i < NUM_SAMPLES; i++) {
    float pred = y_pred.data[i];
    float true_val = Y_data[i];
    total_mae += fabsf(pred - true_val);
  }
  float mae = total_mae / NUM_SAMPLES;
  log_info("Mean Absolute Error: %.4f (normalized), %.4f (in $1000s)", mae,
           mae * 55.0f);

  // Print sample predictions
  log_info("Sample predictions (first 5):");
  for (size_t i = 0; i < 5 && i < NUM_SAMPLES; i++) {
    float pred = y_pred.data[i];
    float true_val = Y_data[i];
    float error = (pred - true_val) * 55.0f; // Denormalize for display
    float pred_price = pred * 55.0f;
    float true_price = true_val * 55.0f;
    log_info("  Sample %zu: Predicted: $%.1fk, True: $%.1fk, Error: $%.2fk", i,
             pred_price, true_price, error);
  }

  nm_tensor_free(&y_pred);
  nm_network_free(net);

  return EXIT_SUCCESS;
}
