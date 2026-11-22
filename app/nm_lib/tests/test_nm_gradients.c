/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Gradient check: for every parameter sampled from a set of small models, the
 * gradient the trainer applies must match a numerical derivative of the loss.
 *
 * With plain SGD and one full batch, one training step moves a weight by
 * lr * gradient, so (before - after) / lr is the analytic gradient. The numeric
 * gradient comes from forward passes with the weight nudged up and down. New
 * layers or operations only need a model entry in MODELS. */
#include "nm_engine_c.h"
#include "nm_graph_ir_internal.h"
#include "nm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNITY_INCLUDE_FLOAT
#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

typedef struct {
  const char *name;
  NmLoss loss;
  size_t input_cols; /* flattened input */
  size_t outputs;
  NmFrontendModel *(*build)(void);
} ModelCase;

static void must(NmStatus status) { TEST_ASSERT_EQUAL(NM_STATUS_OK, status); }

static NmFrontendModel *dense_tanh_linear(void) {
  nm_set_seed(11);
  NmFrontendModel *m = nm_frontend_create((NmShape){.ndim = 1, .dims = {5}});
  must(nm_frontend_add_dense(m, &(NmDenseConfig){.units = 4, .activation = NM_ACT_TANH}));
  must(nm_frontend_add_dense(m, &(NmDenseConfig){.units = 3, .activation = NM_ACT_NONE}));
  must(nm_frontend_finalize(m));
  return m;
}

static NmFrontendModel *dense_sigmoid_softmax(void) {
  nm_set_seed(12);
  NmFrontendModel *m = nm_frontend_create((NmShape){.ndim = 1, .dims = {5}});
  must(nm_frontend_add_dense(m, &(NmDenseConfig){.units = 4, .activation = NM_ACT_SIGMOID}));
  must(nm_frontend_add_dense(m, &(NmDenseConfig){.units = 3, .activation = NM_ACT_SOFTMAX}));
  must(nm_frontend_finalize(m));
  return m;
}

static NmFrontendModel *conv_pool_dense(void) {
  nm_set_seed(13);
  NmFrontendModel *m = nm_frontend_create((NmShape){.ndim = 3, .dims = {1, 6, 6}});
  must(nm_frontend_add_conv2d(m, &(NmConv2dConfig){.filters = 2, .kernel_size = 3, .stride = 1,
                                                   .padding = 1, .activation = NM_ACT_TANH}));
  must(nm_frontend_add_maxpool2d(m, &(NmPool2dConfig){.kernel_size = 2, .stride = 2}));
  must(nm_frontend_add_flatten(m));
  must(nm_frontend_add_dense(m, &(NmDenseConfig){.units = 3, .activation = NM_ACT_SOFTMAX}));
  must(nm_frontend_finalize(m));
  return m;
}

static NmFrontendModel *conv_stride2_dense(void) {
  nm_set_seed(14);
  NmFrontendModel *m = nm_frontend_create((NmShape){.ndim = 3, .dims = {2, 7, 7}});
  must(nm_frontend_add_conv2d(m, &(NmConv2dConfig){.filters = 3, .kernel_size = 3, .stride = 2,
                                                   .padding = 0, .activation = NM_ACT_NONE}));
  must(nm_frontend_add_flatten(m));
  must(nm_frontend_add_dense(m, &(NmDenseConfig){.units = 2, .activation = NM_ACT_TANH}));
  must(nm_frontend_finalize(m));
  return m;
}

static const ModelCase MODELS[] = {
    {"dense tanh + linear, MSE", NM_LOSS_MSE, 5, 3, dense_tanh_linear},
    {"dense sigmoid + softmax, cross entropy", NM_LOSS_CROSS_ENTROPY, 5, 3, dense_sigmoid_softmax},
    {"conv tanh + maxpool + flatten + softmax, cross entropy", NM_LOSS_CROSS_ENTROPY, 36, 3,
     conv_pool_dense},
    {"conv stride 2 + flatten + dense tanh, MSE", NM_LOSS_MSE, 98, 2, conv_stride2_dense},
};

/* Loss as the trainer differentiates it: sum over outputs, mean over the batch. */
static double loss_value(NmLoss loss, const FloatMatrix *pred, const FloatMatrix *target) {
  double total = 0;
  for (size_t i = 0; i < pred->rows * pred->cols; ++i) {
    if (loss == NM_LOSS_MSE) {
      double d = (double)pred->values[i] - (double)target->values[i];
      total += 0.5 * d * d;
    } else {
      total -= (double)target->values[i] * log((double)pred->values[i] + 1e-12);
    }
  }
  return total / (double)pred->rows;
}

static double forward_loss(NmEngineC *engine, NmFrontendModel *model, const FloatMatrix *x,
                           const FloatMatrix *y, NmLoss loss) {
  FloatMatrix *pred = NULL;
  must(nm_engine_c_predict(engine, model, x, &pred));
  double value = loss_value(loss, pred, y);
  sm_destroy(pred);
  return value;
}

static void check_model(const ModelCase *c) {
  const size_t rows = 6;
  NmFrontendModel *model = c->build();
  NmIrGraph *graph = nm_frontend_graph(model);

  NmEngineCConfig config = nm_engine_c_defaults();
  config.loss_type = c->loss;
  config.optimizer.type = NM_OPT_SGD;
  NmEngineC *engine = nm_engine_c_create(&config);
  must(nm_engine_c_prepare(engine, model));

  FloatMatrix *x = sm_create(rows, c->input_cols), *y = sm_create(rows, c->outputs);
  for (size_t i = 0; i < rows * c->input_cols; ++i)
    x->values[i] = (float)((int)((i * 7 + 3) % 13) - 6) / 6.0f;
  for (size_t r = 0; r < rows; ++r) {
    for (size_t k = 0; k < c->outputs; ++k)
      y->values[r * c->outputs + k] =
          c->loss == NM_LOSS_CROSS_ENTROPY ? (k == r % c->outputs ? 1.0f : 0.0f)
                                           : (float)((int)((r + 2 * k) % 5) - 2) * 0.3f;
  }

  /* Snapshot every constant, then take one SGD step on the full batch. */
  const float lr = 1e-2f;
  float *before[32] = {0};
  size_t counts[32] = {0};
  TEST_ASSERT_TRUE(graph->value_count <= 32);
  for (size_t v = 0; v < graph->value_count; ++v) {
    const NmIrValue *value = &graph->values[v];
    if (value->kind != NM_IR_VALUE_CONSTANT)
      continue;
    counts[v] = value->data_count;
    before[v] = malloc(counts[v] * sizeof(float));
    memcpy(before[v], value->data, counts[v] * sizeof(float));
  }
  NmTrainConfig step = {.batch_size = rows, .learning_rate = lr, .epochs = 1};
  must(nm_engine_c_fit(engine, model, x, y, &step));
  float *after[32] = {0};
  for (size_t v = 0; v < graph->value_count; ++v) {
    if (!before[v])
      continue;
    after[v] = malloc(counts[v] * sizeof(float));
    memcpy(after[v], graph->values[v].data, counts[v] * sizeof(float));
    memcpy(graph->values[v].data, before[v], counts[v] * sizeof(float)); /* restore */
  }

  size_t checked = 0;
  for (size_t v = 0; v < graph->value_count; ++v) {
    if (!before[v])
      continue;
    float *w = graph->values[v].data;
    const size_t stride = counts[v] > 8 ? counts[v] / 8 : 1;
    for (size_t i = 0; i < counts[v]; i += stride) {
      const double analytic = ((double)before[v][i] - (double)after[v][i]) / (double)lr;
      const float eps = 5e-3f;
      w[i] = before[v][i] + eps;
      double up = forward_loss(engine, model, x, y, c->loss);
      w[i] = before[v][i] - eps;
      double down = forward_loss(engine, model, x, y, c->loss);
      w[i] = before[v][i];
      const double numeric = (up - down) / (2.0 * (double)eps);
      char message[200];
      snprintf(message, sizeof(message), "%s: %s[%zu] analytic %.6f numeric %.6f", c->name,
               graph->values[v].name, i, analytic, numeric);
      TEST_ASSERT_TRUE_MESSAGE(fabs(analytic - numeric) <= 2e-3 + 3e-2 * fabs(numeric), message);
      ++checked;
    }
  }
  TEST_ASSERT_TRUE_MESSAGE(checked >= 8, c->name);

  for (size_t v = 0; v < graph->value_count; ++v) {
    free(before[v]);
    free(after[v]);
  }
  sm_destroy(y);
  sm_destroy(x);
  nm_engine_c_free(engine);
  nm_frontend_free(model);
}

void test_gradients_match_numerical_derivatives(void) {
  for (size_t i = 0; i < sizeof(MODELS) / sizeof(MODELS[0]); ++i)
    check_model(&MODELS[i]);
}
