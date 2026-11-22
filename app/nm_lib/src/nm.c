/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 *
 * nm.c — Public utility and math helpers for the unified nm API
 *
 * Contents:
 * - default config constructors shared across the library
 * - backend/seed utility functions
 * - activation and activation-gradient helpers
 * - loss functions and reduction helpers
 * - basic matrix-oriented primitives such as linear layers and accuracy
 *
 * The higher-level model implementation lives in `nm_model.c`, while this
 * file provides the reusable low-level helpers that remain part of `nm.h`.
 */

#include "nm.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void sm_set_random_seed(uint64_t seed);

#if defined(USE_ACCELERATE)
#define ACTIVE_LIB "Apple Accelerate"
#define NM_BACKEND_DETERMINISTIC true
#define NM_BACKEND_SUPPORTS_CONV_GEMM true
#elif defined(USE_ACCELERATE_MPS)
#define ACTIVE_LIB "Metal Performance Shaders"
#define NM_BACKEND_DETERMINISTIC false
#define NM_BACKEND_SUPPORTS_CONV_GEMM true
#elif defined(USE_OPENBLAS)
#define ACTIVE_LIB "OpenBLAS"
#define NM_BACKEND_DETERMINISTIC true
#define NM_BACKEND_SUPPORTS_CONV_GEMM true
#else
#define ACTIVE_LIB "No BLAS"
#define NM_BACKEND_DETERMINISTIC true
#define NM_BACKEND_SUPPORTS_CONV_GEMM false
#endif

static const NmBackendCapabilities NM_ACTIVE_BACKEND_CAPS = {
  .name = ACTIVE_LIB,
  .deterministic = NM_BACKEND_DETERMINISTIC,
  .supports_training = true,
  .supports_autograd_tape = true,
  .supports_tensor_pool_reuse = true,
  .supports_prepared_inference = true,
  .supports_fused_dense_relu = true,
  .supports_conv_gemm = NM_BACKEND_SUPPORTS_CONV_GEMM,
  .prefers_prepared_inference = true,
};

static size_t matrix_numel(const FloatMatrix *m) {
  return m ? m->rows * m->cols : 0;
}

static bool same_shape(const FloatMatrix *a, const FloatMatrix *b) {
  return a && b && a->rows == b->rows && a->cols == b->cols;
}

static size_t argmax_row_index(const FloatMatrix *m, size_t row) {
  size_t best = 0;
  float best_val = m->values[row * m->cols];
  for (size_t col = 1; col < m->cols; ++col) {
    float v = m->values[row * m->cols + col];
    if (v > best_val) {
      best_val = v;
      best = col;
    }
  }
  return best;
}

NmCompileConfig nm_compile_defaults(void) {
  NmCompileConfig cfg;
  cfg.loss_type = NM_LOSS_CROSS_ENTROPY;
  return cfg;
}

NmTrainConfig nm_train_defaults(void) {
  NmTrainConfig cfg;
  cfg.batch_size = 32;
  cfg.learning_rate = 0.01f;
  cfg.epochs = 1;
  cfg.verbose = false;
  return cfg;
}

const char *nm_backend(void) { return ACTIVE_LIB; }

const NmBackendCapabilities *nm_backend_capabilities(void) {
  return &NM_ACTIVE_BACKEND_CAPS;
}

void nm_set_seed(uint64_t seed) { sm_set_random_seed(seed); }

void nm_relu(FloatMatrix *mat) {
  sm_apply_relu(mat);
}

void nm_sigmoid(FloatMatrix *mat) {
  sm_apply_sigmoid(mat);
}

float nm_tanhf(float x) { return tanhf(x); }

void nm_tanh(FloatMatrix *mat) {
  sm_apply_tanh(mat);
}

float nm_softmax_denom(const float *vec, size_t len) {
  if (!vec || len == 0)
    return 0.0f;

  float max_val = vec[0];
  for (size_t i = 1; i < len; ++i) {
    if (vec[i] > max_val)
      max_val = vec[i];
  }

  float sum = 0.0f;
  for (size_t i = 0; i < len; ++i) {
    sum += expf(vec[i] - max_val);
  }
  return sum;
}

void nm_softmax(FloatMatrix *mat) {
  sm_apply_softmax(mat);
}

void nm_relu_grad(const FloatMatrix *activation, FloatMatrix *grad_output) {
  sm_apply_relu_backward(activation, grad_output);
}

void nm_sigmoid_grad(const FloatMatrix *activation, FloatMatrix *grad_output) {
  sm_apply_sigmoid_backward(activation, grad_output);
}

void nm_tanh_grad(const FloatMatrix *activation, FloatMatrix *grad_output) {
  sm_apply_tanh_backward(activation, grad_output);
}

void nm_softmax_ce_grad(const FloatMatrix *predicted,
                        const FloatMatrix *target,
                        FloatMatrix *grad_output) {
  sm_softmax_ce_grad(predicted, target, grad_output);
}

float nm_loss_mse(const FloatMatrix *predicted, const FloatMatrix *target) {
  return sm_loss_mse(predicted, target);
}

float nm_loss_cross_entropy(const FloatMatrix *predicted,
                            const FloatMatrix *target) {
  return sm_loss_cross_entropy(predicted, target);
}

void nm_add_rowwise(FloatMatrix *mat, const FloatMatrix *row) {
  sm_add_rowwise(mat, row);
}

NmStatus nm_sum_rows(const FloatMatrix *mat, FloatMatrix **out_sum_rows) {
  if (!mat || !out_sum_rows)
    return NM_STATUS_INVALID_ARGUMENT;
  *out_sum_rows = sm_sum_rows(mat);
  return *out_sum_rows ? NM_STATUS_OK : NM_STATUS_ALLOC_FAILED;
}

NmStatus nm_argmax_rows(const FloatMatrix *mat, FloatMatrix **out_argmax) {
  if (!mat || !out_argmax || mat->cols == 0)
    return NM_STATUS_INVALID_ARGUMENT;
  *out_argmax = sm_argmax_rows(mat);
  return *out_argmax ? NM_STATUS_OK : NM_STATUS_ALLOC_FAILED;
}

NmStatus nm_accuracy(const FloatMatrix *predicted,
                     const FloatMatrix *target,
                     float *out_accuracy) {
  if (!same_shape(predicted, target) || !out_accuracy || predicted->rows == 0 ||
      predicted->cols == 0) {
    return NM_STATUS_INVALID_ARGUMENT;
  }

  size_t correct = 0;
  for (size_t r = 0; r < predicted->rows; ++r) {
    if (argmax_row_index(predicted, r) == argmax_row_index(target, r)) {
      ++correct;
    }
  }

  *out_accuracy = (float)correct / (float)predicted->rows;
  return NM_STATUS_OK;
}

NmStatus nm_linear_into(const FloatMatrix *input, const FloatMatrix *weights,
                        const FloatMatrix *bias, FloatMatrix *out_result) {
  if (!input || !weights || !bias || !out_result || !input->values ||
      !weights->values || !bias->values || !out_result->values) {
    return NM_STATUS_INVALID_ARGUMENT;
  }
  if (input->cols != weights->rows || bias->rows != 1 ||
      bias->cols != weights->cols || out_result->rows != input->rows ||
      out_result->cols != weights->cols) {
    return NM_STATUS_INVALID_ARGUMENT;
  }

  if (!sm_gemm(out_result, 1.0f, input, SM_NO_TRANSPOSE, weights,
               SM_NO_TRANSPOSE, 0.0f)) {
    return NM_STATUS_INTERNAL_ERROR;
  }

  nm_add_rowwise(out_result, bias);
  return NM_STATUS_OK;
}

NmStatus nm_linear(const FloatMatrix *input, const FloatMatrix *weights,
                   const FloatMatrix *bias, FloatMatrix **out_result) {
  if (!input || !weights || !bias || !out_result)
    return NM_STATUS_INVALID_ARGUMENT;
  if (input->cols != weights->rows || bias->rows != 1 ||
      bias->cols != weights->cols) {
    return NM_STATUS_INVALID_ARGUMENT;
  }

  *out_result = sm_create(input->rows, weights->cols);
  if (!*out_result)
    return NM_STATUS_ALLOC_FAILED;

  NmStatus status = nm_linear_into(input, weights, bias, *out_result);
  if (status != NM_STATUS_OK) {
    sm_destroy(*out_result);
    *out_result = NULL;
  }
  return status;
}
