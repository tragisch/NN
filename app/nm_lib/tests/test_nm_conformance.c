/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Conformance with GraphIR: runs the shared corpus and checks that the
 * published capability profile says what nm_lib really does. */
#include "nm_capabilities.h"
#include "nm_engine_c.h"
#include "nm_import_graphir.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNITY_INCLUDE_FLOAT
#define TEST_CASE(...)
#include "unity.h"

#define DIR "app/nm_lib/tests/fixtures/conformance/"

void setUp(void) {}
void tearDown(void) {}

static char *read_text(const char *path) {
  FILE *f = fopen(path, "rb");
  TEST_ASSERT_NOT_NULL_MESSAGE(f, path);
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *text = calloc((size_t)n + 1, 1);
  TEST_ASSERT_EQUAL_size_t((size_t)n, fread(text, 1, (size_t)n, f));
  fclose(f);
  return text;
}

static void *read_bytes(const char *file, size_t *bytes) {
  char path[512];
  snprintf(path, sizeof(path), DIR "%s", file);
  FILE *f = fopen(path, "rb");
  TEST_ASSERT_NOT_NULL_MESSAGE(f, path);
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  void *values = malloc(n ? (size_t)n : 1);
  TEST_ASSERT_EQUAL_size_t((size_t)n, fread(values, 1, (size_t)n, f));
  fclose(f);
  *bytes = (size_t)n;
  return values;
}

static float *read_floats(const char *file, size_t *count) {
  size_t bytes = 0;
  float *values = read_bytes(file, &bytes);
  *count = bytes / sizeof(float);
  return values;
}

/* Value of "key": "..." on one manifest line, copied into `out`. */
static int field(const char *line, const char *key, char *out, size_t size) {
  char needle[64];
  snprintf(needle, sizeof(needle), "\"%s\": \"", key);
  const char *p = strstr(line, needle);
  if (!p)
    return 0;
  p += strlen(needle);
  const char *end = strchr(p, '"');
  TEST_ASSERT_NOT_NULL(end);
  size_t n = (size_t)(end - p);
  TEST_ASSERT_TRUE(n < size);
  memcpy(out, p, n);
  out[n] = '\0';
  return 1;
}

/* Number after "key": on a manifest line, or `fallback` if the key is absent. */
static double number_field(const char *line, const char *key, double fallback) {
  char needle[64];
  snprintf(needle, sizeof(needle), "\"%s\": ", key);
  const char *p = strstr(line, needle);
  return p ? atof(p + strlen(needle)) : fallback;
}

static size_t g_supported, g_unsupported, g_typed;

/* Reads the typed inputs of a manifest line: {"name": ..., "dtype": ...,
 * "shape": [...], "file": ...} entries after "inputs": [. Returns the count. */
static size_t typed_inputs(const char *line, NmInput *inputs, char names[][64],
                           void **buffers) {
  const char *p = strstr(line, "\"inputs\": [");
  if (!p)
    return 0;
  size_t count = 0;
  while ((p = strstr(p, "{\"name\": \"")) != NULL) {
    TEST_ASSERT_TRUE(count < 8);
    char dtype[16], file[160];
    TEST_ASSERT_TRUE(field(p, "name", names[count], 64));
    TEST_ASSERT_TRUE(field(p, "dtype", dtype, sizeof(dtype)));
    TEST_ASSERT_TRUE(field(p, "file", file, sizeof(file)));
    NmInput *input = &inputs[count];
    memset(input, 0, sizeof(*input));
    input->name = names[count];
    input->dtype = !strcmp(dtype, "int64")  ? NM_DTYPE_INT64
                   : !strcmp(dtype, "bool") ? NM_DTYPE_BOOL
                                            : NM_DTYPE_FLOAT32;
    const char *shape = strstr(p, "\"shape\": [") + strlen("\"shape\": [");
    while (*shape != ']') {
      char *end = NULL;
      input->dims[input->ndim++] = (size_t)strtoul(shape, &end, 10);
      shape = end;
      while (*shape == ',' || *shape == ' ')
        shape++;
    }
    size_t bytes = 0;
    buffers[count] = read_bytes(file, &bytes);
    input->data = buffers[count];
    count++;
    p = strstr(p, "}") + 1;
  }
  return count;
}

static void run_case(const char *line) {
  char name[128], support[32], graph_file[160], input_file[160], expected_file[160];
  TEST_ASSERT_TRUE(field(line, "name", name, sizeof(name)));
  TEST_ASSERT_TRUE(field(line, "support", support, sizeof(support)));
  TEST_ASSERT_TRUE(field(line, "graph", graph_file, sizeof(graph_file)));
  char path[512];
  snprintf(path, sizeof(path), DIR "%s", graph_file);

  NmGraphModel *graph = NULL;
  NmEngineC *engine = nm_engine_c_create(NULL);
  NmStatus import = nm_import_graphir_json(path, &graph);
  NmStatus prepare = import == NM_STATUS_OK ? nm_engine_c_prepare_graph(engine, graph)
                                            : import;

  if (!strcmp(support, "unsupported")) {
    TEST_ASSERT_EQUAL_MESSAGE(NM_STATUS_UNSUPPORTED_TYPE, prepare, name);
    ++g_unsupported;
  } else {
    TEST_ASSERT_EQUAL_MESSAGE(NM_STATUS_OK, import, name);
    TEST_ASSERT_EQUAL_MESSAGE(NM_STATUS_OK, prepare, name);
    TEST_ASSERT_TRUE(field(line, "expected", expected_file, sizeof(expected_file)));
    size_t in_count = 0, out_count = 0;
    float *expected = read_floats(expected_file, &out_count);
    /* 16-bit graphs run promoted in f32, so they differ by rounding. */
    const double tolerance = number_field(line, "tolerance", 1e-4);
    NmInput inputs[8];
    char names[8][64];
    void *buffers[8] = {NULL};
    size_t typed = typed_inputs(line, inputs, names, buffers);
    float *input = NULL;
    FloatMatrix *x = NULL;
    FloatMatrix *y = NULL;
    if (typed) {
      /* Token ids, masks and several inputs go through the named API. */
      TEST_ASSERT_EQUAL_MESSAGE(NM_STATUS_OK,
                                nm_engine_c_predict_inputs(engine, graph, inputs, typed, &y),
                                name);
      ++g_typed;
    } else {
      TEST_ASSERT_TRUE(field(line, "input", input_file, sizeof(input_file)));
      input = read_floats(input_file, &in_count);
      const size_t batch = 2;
      x = sm_create(batch, in_count / batch);
      memcpy(x->values, input, in_count * sizeof(float));
      TEST_ASSERT_EQUAL_MESSAGE(NM_STATUS_OK, nm_engine_c_predict_graph(engine, graph, x, &y),
                                name);
    }
    TEST_ASSERT_EQUAL_size_t_MESSAGE(out_count, y->rows * y->cols, name);
    for (size_t i = 0; i < out_count; ++i)
      TEST_ASSERT_TRUE_MESSAGE(fabs((double)expected[i] - (double)y->values[i]) <= tolerance,
                               name);
    sm_destroy(y);
    if (x)
      sm_destroy(x);
    for (size_t i = 0; i < typed; ++i)
      free(buffers[i]);
    free(expected);
    free(input);
    ++g_supported;
  }
  nm_graph_model_free(graph);
  nm_engine_c_free(engine);
}

void test_every_corpus_case_behaves_as_the_manifest_says(void) {
  char *manifest = read_text(DIR "manifest.json");
  for (char *line = strtok(manifest, "\n"); line; line = strtok(NULL, "\n")) {
    if (strstr(line, "\"name\":"))
      run_case(line);
  }
  free(manifest);
  TEST_ASSERT_TRUE(g_supported >= 9);
  TEST_ASSERT_TRUE(g_unsupported >= 9);
  TEST_ASSERT_TRUE(g_typed >= 3);
}

void test_published_profile_matches_the_copy_graphir_uses(void) {
  char *committed = read_text("app/nm_lib/tests/fixtures/nm_lib.profile.json");
  TEST_ASSERT_EQUAL_STRING(nm_capabilities_json(), committed);
  free(committed);
}

/* Every operation the profile lists must be one the importer knows, and the
 * corpus must exercise each of them in a supported case. */
void test_every_listed_operation_appears_in_a_supported_case(void) {
  const char *profile = nm_capabilities_json();
  char *manifest = read_text(DIR "manifest.json");
  static char all_graphs[1 << 18];
  all_graphs[0] = '\0';
  for (char *line = strtok(manifest, "\n"); line; line = strtok(NULL, "\n")) {
    char support[32], graph_file[160];
    if (field(line, "support", support, sizeof(support)) && !strcmp(support, "supported") &&
        field(line, "graph", graph_file, sizeof(graph_file))) {
      char path[512];
      snprintf(path, sizeof(path), DIR "%s", graph_file);
      char *text = read_text(path);
      TEST_ASSERT_TRUE(strlen(all_graphs) + strlen(text) < sizeof(all_graphs));
      strcat(all_graphs, text);
      free(text);
    }
  }
  free(manifest);
  for (const char *p = strstr(profile, "{\"name\": \""); p; p = strstr(p + 1, "{\"name\": \"")) {
    p += strlen("{\"name\": \"");
    char op[64];
    const char *end = strchr(p, '"');
    memcpy(op, p, (size_t)(end - p));
    op[end - p] = '\0';
    char needle[96];
    snprintf(needle, sizeof(needle), "\"operation\":\"%s\"", op);
    TEST_ASSERT_NOT_NULL_MESSAGE(strstr(all_graphs, needle), op);
  }
}
