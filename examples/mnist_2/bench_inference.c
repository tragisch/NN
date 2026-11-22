/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Pure inference benchmark: load GraphIR JSON + binary test inputs,
 * measure nm_engine_c_predict_graph latency. No training.
 *
 * Build (from repo root):
 *   clang -O2 -std=c11 examples/mnist_2/bench_inference.c \
 *     -Iapp/nm_lib/include -Imatrix/app/matrix/include \
 *     bazel-bin/app/nm_lib/ffi/libnm_api_static.a \
 *     -framework Accelerate -framework Foundation \
 *     -o bazel-bin/bench_inference
 */
#include "nm_engine_c.h"
#include "nm_import_graphir.h"
#include "sm.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
  const char *graph_path = argc > 1 ? argv[1] : "mnist.graphir.json";
  const char *inputs_path = argc > 2 ? argv[2] : "mnist_test_inputs.bin";
  const char *labels_path = argc > 3 ? argv[3] : "mnist_test_labels.bin";
  int iters = argc > 4 ? atoi(argv[4]) : 10;

  /* Load graph */
  NmGraphModel *graph = NULL;
  NmStatus s = nm_import_graphir_json(graph_path, &graph);
  if (s != NM_STATUS_OK || !graph) {
    fprintf(stderr, "Failed to import %s (status=%d)\n", graph_path, (int)s);
    return 1;
  }

  NmEngineC *engine = nm_engine_c_create(NULL);
  s = nm_engine_c_prepare_graph(engine, graph);
  if (s != NM_STATUS_OK) {
    fprintf(stderr, "Failed to prepare graph (status=%d)\n", (int)s);
    return 1;
  }

  /* Load binary inputs: 1000 x 784 float32 */
  FILE *f = fopen(inputs_path, "rb");
  if (!f) { perror("inputs"); return 1; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  float *data = malloc(sz);
  fread(data, 1, sz, f);
  fclose(f);
  size_t n_samples = sz / (784 * sizeof(float));
  size_t input_dim = 784;

  FloatMatrix *X = sm_create_with_values(n_samples, input_dim, data);

  /* Load labels */
  f = fopen(labels_path, "rb");
  if (!f) { perror("labels"); return 1; }
  fseek(f, 0, SEEK_END);
  long lsz = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *labels = malloc(lsz);
  fread(labels, 1, lsz, f);
  fclose(f);

  /* Warmup + benchmark */
  FloatMatrix *pred = NULL;
  s = nm_engine_c_predict_graph(engine, graph, X, &pred);
  if (s != NM_STATUS_OK || !pred) {
    fprintf(stderr, "Predict failed (status=%d)\n", (int)s);
    return 1;
  }

  /* Accuracy */
  size_t correct = 0;
  for (size_t i = 0; i < n_samples && i < (size_t)lsz; i++) {
    size_t best = 0;
    for (size_t c = 1; c < 10; c++)
      if (pred->values[i * 10 + c] > pred->values[i * 10 + best]) best = c;
    if (best == labels[i]) correct++;
  }
  printf("Accuracy: %zu/%zu = %.2f%%\n", correct, n_samples, 100.0 * correct / n_samples);

  /* Timed iterations */
  double total = 0.0;
  for (int i = 0; i < iters; i++) {
    FloatMatrix *tmp = NULL;
    double t0 = now_ms();
    nm_engine_c_predict_graph(engine, graph, X, &tmp);
    total += now_ms() - t0;
    if (tmp) sm_destroy(tmp);
  }
  double mean = total / iters;
  printf("Mean latency: %.3f ms  (%.4f ms/sample, %.1f samples/s)\n",
         mean, mean / n_samples, n_samples * iters / (total / 1000.0));

  sm_destroy(pred);
  sm_destroy(X);
  free(data);
  free(labels);
  nm_engine_c_free(engine);
  nm_graph_model_free(graph);
  return 0;
}
