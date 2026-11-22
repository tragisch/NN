/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* A convolution graph that is not a layer chain runs on the interpreter, which
 * hands Conv2D and MaxPool2D to the mMatrix kernels. The graph is the
 * conformance case conv_relu_pool_dense followed by a multiplication with 1.0,
 * so its input and expected output are the conformance ones. */
#include "nm_engine_c.h"
#include "nm_import_graphir.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define UNITY_INCLUDE_FLOAT
#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static float *read_floats(const char *path, size_t *count) {
  FILE *f = fopen(path, "rb");
  TEST_ASSERT_NOT_NULL_MESSAGE(f, path);
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  float *values = malloc((size_t)n);
  TEST_ASSERT_EQUAL_size_t((size_t)n, fread(values, 1, (size_t)n, f));
  fclose(f);
  *count = (size_t)n / sizeof(float);
  return values;
}

void test_conv_graph_beyond_a_chain_matches_the_reference_output(void) {
  const char *dir = "app/nm_lib/tests/fixtures/";
  NmGraphModel *graph = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_import_graphir_json("app/nm_lib/tests/fixtures/interp/"
                                           "conv_relu_pool_dense_scaled.graphir.json",
                                           &graph));
  NmEngineC *engine = nm_engine_c_create(NULL);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare_graph(engine, graph));

  char path[256];
  size_t in_count = 0, out_count = 0;
  snprintf(path, sizeof(path), "%sconformance/conv_relu_pool_dense.input.f32", dir);
  float *input = read_floats(path, &in_count);
  snprintf(path, sizeof(path), "%sconformance/conv_relu_pool_dense.expected.f32", dir);
  float *expected = read_floats(path, &out_count);

  /* two examples of 1 x 6 x 6 values */
  FloatMatrix x = {.rows = 2, .cols = in_count / 2, .capacity = in_count,
                   .values = input};
  FloatMatrix *y = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_predict_graph(engine, graph, &x, &y));
  TEST_ASSERT_EQUAL_size_t(out_count, y->rows * y->cols);
  for (size_t i = 0; i < out_count; ++i)
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected[i], y->values[i]);

  /* Only the interpreter refuses training, so this proves which path ran. */
  NmTrainConfig train = {.batch_size = 2, .learning_rate = 0.01f, .epochs = 1};
  TEST_ASSERT_NOT_EQUAL(NM_STATUS_OK,
                        nm_engine_c_fit_graph(engine, graph, &x, y, &train));

  sm_destroy(y);
  free(input);
  free(expected);
  nm_engine_c_free(engine);
  nm_graph_model_free(graph);
}
