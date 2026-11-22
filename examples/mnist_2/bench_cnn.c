/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* MNIST CNN via frontend+engine: Conv2D->ReLU->MaxPool->Conv2D->ReLU->MaxPool->Flatten->Dense->Softmax
 * Exports GraphIR JSON, benchmarks C engine inference.
 *
 * clang -O2 -std=c11 examples/mnist_2/bench_cnn.c \
 *   -Iapp/nm_lib/include -I<mmatrix>/app/matrix/include \
 *   bazel-bin/app/nm_lib/ffi/libnm_api_static.a \
 *   -framework Accelerate -framework Metal -framework MetalPerformanceShaders \
 *   -framework MetalPerformanceShadersGraph -framework Foundation -o /tmp/bench_cnn
 */
#include "nm_engine_c.h"
#include "nm_export_graphir.h"
#include "nm_frontend.h"
#include "sm.h"
#include "mnist_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
  /* Input: batch 100, 1 channel, 28x28 */
  NmFrontendModel *model = nm_frontend_create((NmShape){.ndim = 3, .dims = {1, 28, 28}});
  NmStatus s = model ? NM_STATUS_OK : NM_STATUS_ALLOC_FAILED;

  if (s == NM_STATUS_OK)
    s = nm_frontend_add_conv2d(model, &(NmConv2dConfig){
        .filters = 8, .kernel_size = 3, .stride = 1, .padding = 1,
        .activation = NM_ACT_RELU});
  if (s == NM_STATUS_OK)
    s = nm_frontend_add_maxpool2d(model, &(NmPool2dConfig){.kernel_size = 2, .stride = 2, .padding = 0});
  if (s == NM_STATUS_OK)
    s = nm_frontend_add_conv2d(model, &(NmConv2dConfig){
        .filters = 16, .kernel_size = 3, .stride = 1, .padding = 1,
        .activation = NM_ACT_RELU});
  if (s == NM_STATUS_OK)
    s = nm_frontend_add_maxpool2d(model, &(NmPool2dConfig){.kernel_size = 2, .stride = 2, .padding = 0});
  if (s == NM_STATUS_OK)
    s = nm_frontend_add_flatten(model);
  if (s == NM_STATUS_OK)
    s = nm_frontend_add_dense(model, &(NmDenseConfig){.units = 10, .activation = NM_ACT_SOFTMAX});
  if (s == NM_STATUS_OK)
    s = nm_frontend_finalize(model);
  if (s != NM_STATUS_OK) { fprintf(stderr, "frontend failed: %d\n", (int)s); return 1; }

  NmEngineC *engine = nm_engine_c_create(NULL);
  s = nm_engine_c_prepare(engine, model);
  if (s != NM_STATUS_OK) { fprintf(stderr, "prepare failed: %d\n", (int)s); return 1; }

  /* Export GraphIR JSON (batch 100) */
  s = nm_export_graphir_json(model, &(NmGraphIrExportOptions){.batch_size = 100}, "mnist_cnn.graphir.json");
  if (s != NM_STATUS_OK) { fprintf(stderr, "export failed: %d\n", (int)s); return 1; }
  printf("Exported mnist_cnn.graphir.json\n");

  /* Load 100 test images via the IDX loader (data fetched by Bazel) */
  FloatMatrix *X = load_mnist_images("examples/mnist/t10k-images-idx3-ubyte", 100);
  if (!X) { fprintf(stderr, "load images failed\n"); return 1; }
  FloatMatrix *Y = load_mnist_labels_onehot("examples/mnist/t10k-labels-idx1-ubyte", 100, 10);
  if (!Y) { fprintf(stderr, "load labels failed\n"); return 1; }
  uint8_t labels[100];
  for (size_t i = 0; i < 100; i++) {
    size_t best = 0;
    for (size_t c = 1; c < 10; c++)
      if (Y->values[i * 10 + c] > Y->values[i * 10 + best]) best = c;
    labels[i] = (uint8_t)best;
  }
  FloatMatrix *pred = NULL;
  s = nm_engine_c_predict(engine, model, X, &pred);
  if (s != NM_STATUS_OK || !pred) { fprintf(stderr, "predict failed: %d\n", (int)s); return 1; }

  size_t correct = 0;
  for (size_t i = 0; i < 100; i++) {
    size_t best = 0;
    for (size_t c = 1; c < 10; c++)
      if (pred->values[i * 10 + c] > pred->values[i * 10 + best]) best = c;
    if (best == labels[i]) correct++;
  }

  /* Benchmark */
  double total = 0;
  for (int i = 0; i < 20; i++) {
    FloatMatrix *tmp = NULL;
    double t0 = now_ms();
    nm_engine_c_predict(engine, model, X, &tmp);
    total += now_ms() - t0;
    sm_destroy(tmp);
  }
  printf("C engine CNN: accuracy=%zu/100=%.0f%%, mean latency=%.3f ms (100 imgs)\n",
         correct, correct, total / 20);

  sm_destroy(pred);
  sm_destroy(X);
  sm_destroy(Y);
  nm_engine_c_free(engine);
  nm_frontend_free(model);
  return 0;
}
