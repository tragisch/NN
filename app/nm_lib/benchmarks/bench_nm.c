/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* nm_lib side of the PyTorch comparison. Prints one JSON object per line:
 * {"name": ..., "median_us": ..., "p90_us": ...}. Same models and data as
 * bench_torch.py. Training steps are one nm_engine_c_fit call over one batch. */
#define _POSIX_C_SOURCE 200809L
#include "nm_engine_c.h"
#include "nm.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BATCH 64
#define WARMUP 30
#define RUNS 200

static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

static void report(const char *name, double *samples, int n) {
  qsort(samples, (size_t)n, sizeof(double), cmp);
  printf("{\"name\": \"%s\", \"median_us\": %.1f, \"p90_us\": %.1f}\n", name, samples[n / 2],
         samples[n * 9 / 10]);
}

static NmFrontendModel *mlp(void) {
  nm_set_seed(1);
  NmFrontendModel *m = nm_frontend_create((NmShape){.ndim = 1, .dims = {784}});
  nm_frontend_add_dense(m, &(NmDenseConfig){.units = 256, .activation = NM_ACT_RELU});
  nm_frontend_add_dense(m, &(NmDenseConfig){.units = 64, .activation = NM_ACT_RELU});
  nm_frontend_add_dense(m, &(NmDenseConfig){.units = 10, .activation = NM_ACT_SOFTMAX});
  nm_frontend_finalize(m);
  return m;
}

static NmFrontendModel *cnn(void) {
  nm_set_seed(1);
  NmFrontendModel *m = nm_frontend_create((NmShape){.ndim = 3, .dims = {1, 28, 28}});
  nm_frontend_add_conv2d(m, &(NmConv2dConfig){.filters = 8, .kernel_size = 3, .stride = 1,
                                              .padding = 1, .activation = NM_ACT_RELU});
  nm_frontend_add_maxpool2d(m, &(NmPool2dConfig){.kernel_size = 2, .stride = 2});
  nm_frontend_add_flatten(m);
  nm_frontend_add_dense(m, &(NmDenseConfig){.units = 10, .activation = NM_ACT_SOFTMAX});
  nm_frontend_finalize(m);
  return m;
}

static void fill(FloatMatrix *x) {
  for (size_t i = 0; i < x->rows * x->cols; ++i)
    x->values[i] = (float)((i * 7) % 13) / 13.0f;
}

static void run(const char *label, NmFrontendModel *(*build)(void)) {
  NmFrontendModel *model = build();
  NmEngineC *engine = nm_engine_c_create(NULL); /* Adam, cross entropy */
  if (nm_engine_c_prepare(engine, model) != NM_STATUS_OK)
    exit(1);
  const size_t cols = 784;

  FloatMatrix *x = sm_create(BATCH, cols), *y = sm_create(BATCH, 10);
  fill(x);
  for (size_t r = 0; r < BATCH; ++r)
    y->values[r * 10 + r % 10] = 1.0f;
  NmTrainConfig step = {.batch_size = BATCH, .learning_rate = 1e-3f, .epochs = 1};

  double samples[RUNS];
  char name[96];
  for (int i = 0; i < WARMUP + RUNS; ++i) {
    double t0 = now_us();
    nm_engine_c_fit(engine, model, x, y, &step);
    double t1 = now_us();
    if (i >= WARMUP)
      samples[i - WARMUP] = t1 - t0;
  }
  snprintf(name, sizeof(name), "%s_train_step_b%d", label, BATCH);
  report(name, samples, RUNS);

  const size_t batches[2] = {1, BATCH};
  for (int k = 0; k < 2; ++k) {
    FloatMatrix *in = sm_create(batches[k], cols);
    fill(in);
    for (int i = 0; i < WARMUP + RUNS; ++i) {
      FloatMatrix *out = NULL;
      double t0 = now_us();
      nm_engine_c_predict(engine, model, in, &out);
      double t1 = now_us();
      sm_destroy(out);
      if (i >= WARMUP)
        samples[i - WARMUP] = t1 - t0;
    }
    snprintf(name, sizeof(name), "%s_infer_b%zu", label, batches[k]);
    report(name, samples, RUNS);
    sm_destroy(in);
  }
  sm_destroy(y);
  sm_destroy(x);
  nm_engine_c_free(engine);
  nm_frontend_free(model);
}

int main(void) {
  run("mlp", mlp);
  run("cnn", cnn);
  return 0;
}
