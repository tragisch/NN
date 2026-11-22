/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#define _POSIX_C_SOURCE 199309L
#include "nm_engine_c.h"
#include "nm_frontend.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int compare_doubles(const void *a, const void *b) {
  double diff = (*(const double *)a - *(const double *)b);
  if (diff < 0) return -1;
  if (diff > 0) return 1;
  return 0;
}

static double get_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}


int main(void) {
  // 1. Define DeepModel architecture in nm_lib
  NmFrontendModel *model = nm_frontend_create((NmShape){.ndim = 1, .dims = {512}});
  if (!model) {
    fprintf(stderr, "Failed to create frontend model\n");
    return 1;
  }

  if (nm_frontend_add_dense(model, &(NmDenseConfig){.units = 1024, .activation = NM_ACT_RELU}) != NM_STATUS_OK ||
      nm_frontend_add_dense(model, &(NmDenseConfig){.units = 1024, .activation = NM_ACT_RELU}) != NM_STATUS_OK ||
      nm_frontend_add_dense(model, &(NmDenseConfig){.units = 512, .activation = NM_ACT_RELU}) != NM_STATUS_OK ||
      nm_frontend_add_dense(model, &(NmDenseConfig){.units = 256, .activation = NM_ACT_RELU}) != NM_STATUS_OK ||
      nm_frontend_add_dense(model, &(NmDenseConfig){.units = 64, .activation = NM_ACT_TANH}) != NM_STATUS_OK ||
      nm_frontend_add_dense(model, &(NmDenseConfig){.units = 10, .activation = NM_ACT_SOFTMAX}) != NM_STATUS_OK ||
      nm_frontend_finalize(model) != NM_STATUS_OK) {
    fprintf(stderr, "Failed to build and finalize model\n");
    nm_frontend_free(model);
    return 1;
  }

  // 2. Prepare Engine
  NmEngineCConfig config = nm_engine_c_defaults();
  NmEngineC *engine = nm_engine_c_create(&config);
  if (!engine || nm_engine_c_prepare(engine, model) != NM_STATUS_OK) {
    fprintf(stderr, "Failed to create or prepare engine\n");
    nm_engine_c_free(engine);
    nm_frontend_free(model);
    return 1;
  }

  // 3. Create Input (Batch=1, Dim=512)
  FloatMatrix *input = sm_create(1, 512);
  if (!input) {
    nm_engine_c_free(engine);
    nm_frontend_free(model);
    return 1;
  }
  for (size_t i = 0; i < 512; ++i) {
    input->values[i] = 1.0f;
  }

  FloatMatrix *prediction = NULL;

  // 4. Warmup (50 iterations)
  for (int i = 0; i < 50; ++i) {
    if (prediction) {
      sm_destroy(prediction);
      prediction = NULL;
    }
    nm_engine_c_predict(engine, model, input, &prediction);
  }

  // 5. Benchmark (200 iterations)
  const int num_iters = 200;
  double *latencies = (double *)malloc((size_t)num_iters * sizeof(double));
  if (!latencies) {
    sm_destroy(prediction);
    sm_destroy(input);
    nm_engine_c_free(engine);
    nm_frontend_free(model);
    return 1;
  }

  for (int i = 0; i < num_iters; ++i) {
    if (prediction) {
      sm_destroy(prediction);
      prediction = NULL;
    }
    double t0 = get_time_ms();
    nm_engine_c_predict(engine, model, input, &prediction);
    double t1 = get_time_ms();
    latencies[i] = t1 - t0;
  }

  qsort(latencies, (size_t)num_iters, sizeof(double), compare_doubles);
  double p50 = latencies[num_iters * 50 / 100];
  double p90 = latencies[num_iters * 90 / 100];
  double p99 = latencies[num_iters * 99 / 100];
  double sum = 0.0;
  for (int i = 0; i < num_iters; ++i) {
    sum += latencies[i];
  }
  double mean = sum / (double)num_iters;

  printf("{\"p50_ms\": %.4f, \"p90_ms\": %.4f, \"p99_ms\": %.4f, \"mean_ms\": %.4f}\n",
         p50, p90, p99, mean);

  free(latencies);
  if (prediction) sm_destroy(prediction);
  sm_destroy(input);
  nm_engine_c_free(engine);
  nm_frontend_free(model);
  return 0;
}
