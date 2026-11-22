/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* User API: build, fit with progress, predict, evaluate, save and load, plus the
 * error messages callers see. */
#include "nm_api.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__clang__)
/* The error tests pass NULL on purpose. */
#pragma clang diagnostic ignored "-Wnonnull"
#endif

#define UNITY_INCLUDE_FLOAT
#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Address of a 2-D tensor built on the spot (valid for the next few calls). */
static const NmTensor *T2(float *data, size_t rows, size_t cols) {
  static NmTensor ring[8];
  static unsigned next;
  NmTensor *slot = &ring[next++ % 8];
  *slot = nm_tensor_2d(data, rows, cols);
  return slot;
}

static char *read_file(const char *path) {
  FILE *f = fopen(path, "rb");
  TEST_ASSERT_NOT_NULL(f);
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *text = calloc((size_t)size + 1, 1);
  TEST_ASSERT_EQUAL_size_t((size_t)size, fread(text, 1, (size_t)size, f));
  fclose(f);
  return text;
}

static uint64_t g_state = 12345;
static float frand(void) {
  g_state ^= g_state >> 12;
  g_state ^= g_state << 25;
  g_state ^= g_state >> 27;
  return (float)((g_state * 0x2545F4914F6CDD1DULL) >> 40) / (float)(1 << 24);
}

/* Two blobs in the plane, one-hot labels. */
static void blobs(size_t n, float *x, float *y) {
  for (size_t i = 0; i < n; ++i) {
    size_t c = i % 2;
    x[i * 2] = (c ? 1.0f : -1.0f) + 0.3f * (frand() - 0.5f);
    x[i * 2 + 1] = (c ? 1.0f : -1.0f) + 0.3f * (frand() - 0.5f);
    y[i * 2] = c == 0;
    y[i * 2 + 1] = c == 1;
  }
}

static NmSpec *classifier_spec(void) {
  NmSpec *spec = nm_spec_new((NmShape){.ndim = 1, .dims = {2}});
  TEST_ASSERT_NOT_NULL(spec);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_spec_dense(spec, 8, NM_ACT_TANH));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_spec_dense(spec, 2, NM_ACT_SOFTMAX));
  return spec;
}

static NmNetwork *build_seeded(const NmSpec *spec, uint64_t seed) {
  NmBuildOptions b = nm_build_options_default();
  b.seed = seed;
  NmNetwork *m = nm_network_build(spec, &b);
  TEST_ASSERT_NOT_NULL_MESSAGE(m, nm_last_error());
  return m;
}

typedef struct {
  size_t calls;
  float first_loss;
  float last_loss;
  size_t stop_after;
  int saw_val;
} Trace;

static int on_progress(const NmProgress *p, void *user) {
  Trace *t = user;
  if (t->calls == 0)
    t->first_loss = p->loss;
  t->last_loss = p->loss;
  t->calls++;
  if (!isnan(p->val_loss))
    t->saw_val = 1;
  return t->stop_after && t->calls >= t->stop_after;
}

void test_classifier_learns_and_reports_progress(void) {
  enum { N = 200 };
  float x[N * 2], y[N * 2];
  blobs(N, x, y);
  NmSpec *spec = classifier_spec();
  NmNetwork *m = build_seeded(spec, 7);

  NmFitOptions f = nm_fit_options_default();
  f.epochs = 30;
  f.batch_size = 20;
  f.learning_rate = 0.2f;
  f.validation_split = 0.2f;
  Trace trace = {0};
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_network_fit(m, nm_tensor_2d(x, N, 2),
                                 &(NmTensor){.data = y, .ndim = 2, .dims = {N, 2}},
                                 &f, on_progress, &trace));
  TEST_ASSERT_EQUAL_size_t(30, trace.calls);
  TEST_ASSERT_TRUE(trace.last_loss < trace.first_loss);
  TEST_ASSERT_TRUE(trace.saw_val);

  NmMetrics r;
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_network_evaluate(m, nm_tensor_2d(x, N, 2),
                                      nm_tensor_2d(y, N, 2),
                                      NM_LOSS_CROSS_ENTROPY, &r));
  TEST_ASSERT_TRUE_MESSAGE(r.accuracy > 0.95f, "accuracy too low");
  nm_network_free(m);
  nm_spec_free(spec);
}

void test_progress_callback_can_stop_training(void) {
  enum { N = 40 };
  float x[N * 2], y[N * 2];
  blobs(N, x, y);
  NmSpec *spec = classifier_spec();
  NmNetwork *m = build_seeded(spec, 7);
  NmFitOptions f = nm_fit_options_default();
  f.epochs = 50;
  Trace trace = {.stop_after = 3};
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_network_fit(m, nm_tensor_2d(x, N, 2), T2(y, N, 2),
                                 &f, on_progress, &trace));
  TEST_ASSERT_EQUAL_size_t(3, trace.calls);
  nm_network_free(m);
  nm_spec_free(spec);
}

void test_same_seed_gives_same_model(void) {
  float x[4] = {0.5f, -0.5f, 1.0f, 2.0f};
  NmSpec *spec = classifier_spec();
  NmNetwork *a = build_seeded(spec, 99);
  NmNetwork *b = build_seeded(spec, 99);
  NmNetwork *c = build_seeded(spec, 100);
  NmTensor pa, pb, pc;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_network_predict(a, nm_tensor_2d(x, 2, 2), &pa));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_network_predict(b, nm_tensor_2d(x, 2, 2), &pb));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_network_predict(c, nm_tensor_2d(x, 2, 2), &pc));
  TEST_ASSERT_EQUAL_size_t(2, pa.dims[0]);
  TEST_ASSERT_EQUAL_size_t(2, pa.dims[1]);
  TEST_ASSERT_EQUAL_MEMORY(pa.data, pb.data, 4 * sizeof(float));
  TEST_ASSERT_TRUE(memcmp(pa.data, pc.data, 4 * sizeof(float)) != 0);
  nm_tensor_free(&pa);
  nm_tensor_free(&pb);
  nm_tensor_free(&pc);
  nm_network_free(a);
  nm_network_free(b);
  nm_network_free(c);
  nm_spec_free(spec);
}

void test_save_and_load_round_trip(void) {
  enum { N = 60 };
  float x[N * 2], y[N * 2];
  blobs(N, x, y);
  NmSpec *spec = classifier_spec();
  NmNetwork *m = build_seeded(spec, 5);
  NmFitOptions f = nm_fit_options_default();
  f.epochs = 5;
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_network_fit(m, nm_tensor_2d(x, N, 2), T2(y, N, 2),
                                 &f, NULL, NULL));
  const char *dir = "tmp_nm_api_model";
  TEST_ASSERT_EQUAL_MESSAGE(NM_STATUS_OK, nm_network_save(m, dir), nm_last_error());
  char *json = read_file("tmp_nm_api_model/model.graphir.json");
  TEST_ASSERT_NOT_NULL(strstr(json, "\"kind\":\"external\""));
  TEST_ASSERT_NOT_NULL(strstr(json, "weights.girw"));
  free(json);
  NmNetwork *loaded = nm_network_load(dir);
  TEST_ASSERT_NOT_NULL_MESSAGE(loaded, nm_last_error());

  NmTensor before, after;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_network_predict(m, nm_tensor_2d(x, N, 2), &before));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_network_predict(loaded, nm_tensor_2d(x, N, 2), &after));
  TEST_ASSERT_EQUAL_MEMORY(before.data, after.data, N * 2 * sizeof(float));

  /* A loaded model can be trained further. */
  TEST_ASSERT_EQUAL_MESSAGE(
      NM_STATUS_OK,
      nm_network_fit(loaded, nm_tensor_2d(x, N, 2), T2(y, N, 2), &f,
                   NULL, NULL),
      nm_last_error());
  nm_tensor_free(&before);
  nm_tensor_free(&after);
  nm_network_free(loaded);
  nm_network_free(m);
  nm_spec_free(spec);
  remove("tmp_nm_api_model/model.graphir.json");
  remove("tmp_nm_api_model/weights.girw");
  remove(dir);
}

void test_autoencoder_reduces_reconstruction_error(void) {
  enum { N = 120, D = 8 };
  float x[N * D];
  for (size_t i = 0; i < N; ++i) {
    float a = frand() - 0.5f, b = frand() - 0.5f;
    for (size_t j = 0; j < D; ++j)
      x[i * D + j] = a * (float)(j + 1) / D + b * (j % 2 ? 1.0f : -1.0f);
  }
  NmSpec *spec = nm_spec_new((NmShape){.ndim = 1, .dims = {D}});
  nm_spec_dense(spec, 3, NM_ACT_TANH);
  nm_spec_dense(spec, D, NM_ACT_NONE);
  NmNetwork *m = build_seeded(spec, 3);
  NmFitOptions f = nm_fit_options_default();
  f.epochs = 150;
  f.batch_size = 20;
  f.learning_rate = 0.05f;
  Trace trace = {0};
  TEST_ASSERT_EQUAL_MESSAGE(
      NM_STATUS_OK,
      nm_network_fit(m, nm_tensor_2d(x, N, D), NULL, &f, on_progress, &trace),
      nm_last_error());
  TEST_ASSERT_TRUE_MESSAGE(trace.last_loss < 0.5f * trace.first_loss,
                           "reconstruction error did not halve");
  nm_network_free(m);
  nm_spec_free(spec);
}

void test_cnn_trains_on_images(void) {
  enum { N = 16 };
  float x[N * 36], y[N * 2];
  for (size_t i = 0; i < N; ++i) {
    size_t c = i % 2;
    for (size_t p = 0; p < 36; ++p)
      x[i * 36 + p] = (p % 6 < 3) == (c == 0) ? 1.0f : 0.0f;
    y[i * 2] = c == 0;
    y[i * 2 + 1] = c == 1;
  }
  NmSpec *spec = nm_spec_new((NmShape){.ndim = 3, .dims = {1, 6, 6}});
  nm_spec_conv2d(spec, 2, 3, 1, 0, NM_ACT_TANH);
  nm_spec_maxpool2d(spec, 2, 2);
  nm_spec_flatten(spec);
  nm_spec_dense(spec, 2, NM_ACT_SOFTMAX);
  NmNetwork *m = build_seeded(spec, 1);
  NmFitOptions f = nm_fit_options_default();
  f.epochs = 20;
  f.batch_size = 8;
  NmTensor xt = {.data = x, .ndim = 4, .dims = {N, 1, 6, 6}};
  TEST_ASSERT_EQUAL_MESSAGE(
      NM_STATUS_OK, nm_network_fit(m, xt, T2(y, N, 2), &f, NULL, NULL),
      nm_last_error());
  NmMetrics r;
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_network_evaluate(m, xt, nm_tensor_2d(y, N, 2),
                                      NM_LOSS_CROSS_ENTROPY, &r));
  TEST_ASSERT_TRUE(r.accuracy > 0.9f);
  nm_network_free(m);
  nm_spec_free(spec);
}

void test_errors_are_reported_with_a_message(void) {
  TEST_ASSERT_NULL(nm_network_build(NULL, NULL));
  TEST_ASSERT_TRUE(strlen(nm_last_error()) > 0);

  NmSpec *empty = nm_spec_new((NmShape){.ndim = 1, .dims = {2}});
  TEST_ASSERT_NULL(nm_network_build(empty, NULL));
  TEST_ASSERT_NOT_NULL(strstr(nm_last_error(), "no layers"));
  nm_spec_free(empty);

  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT, nm_spec_dense(NULL, 4, NM_ACT_RELU));
  NmSpec *spec = classifier_spec();
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT, nm_spec_dense(spec, 0, NM_ACT_RELU));
  NmNetwork *m = build_seeded(spec, 1);

  float x[6] = {0};
  NmTensor out;
  TEST_ASSERT_NOT_EQUAL(NM_STATUS_OK,
                        nm_network_predict(m, nm_tensor_2d(x, 2, 3), &out));
  TEST_ASSERT_TRUE(strlen(nm_last_error()) > 0);
  TEST_ASSERT_NULL(out.data);

  float y[4] = {0};
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT,
                    nm_network_fit(m, nm_tensor_2d(x, 3, 2), T2(y, 2, 2),
                                 NULL, NULL, NULL));
  TEST_ASSERT_NOT_NULL(strstr(nm_last_error(), "targets"));
  NmFitOptions bad = nm_fit_options_default();
  bad.validation_split = 1.0f;
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT,
                    nm_network_fit(m, nm_tensor_2d(x, 3, 2), T2(y, 3, 2),
                                 &bad, NULL, NULL));
  TEST_ASSERT_NULL(nm_network_load("no_such_dir_for_nm_api"));
  TEST_ASSERT_NOT_NULL(strstr(nm_last_error(), "cannot load"));
  nm_network_free(m);
  nm_spec_free(spec);
}

void test_summary_lists_layers_shapes_and_parameters(void) {
  NmSpec *spec = nm_spec_new((NmShape){.ndim = 3, .dims = {1, 6, 6}});
  nm_spec_conv2d(spec, 2, 3, 1, 0, NM_ACT_TANH);
  nm_spec_maxpool2d(spec, 2, 2);
  nm_spec_flatten(spec);
  nm_spec_dense(spec, 2, NM_ACT_SOFTMAX);
  NmNetwork *m = build_seeded(spec, 1);

  /* conv 2*1*3*3 + 2, dense 8*2 + 2 */
  TEST_ASSERT_EQUAL_size_t(38, nm_network_parameter_count(m));
  char text[512];
  size_t need = nm_network_summary(m, text, sizeof(text));
  TEST_ASSERT_TRUE(need > 0 && need < sizeof(text));
  TEST_ASSERT_EQUAL_size_t(strlen(text), need);
  TEST_ASSERT_NOT_NULL(strstr(text, "Conv2D 2@3x3 + Tanh"));
  TEST_ASSERT_NOT_NULL(strstr(text, "[2, 4, 4]"));
  TEST_ASSERT_NOT_NULL(strstr(text, "MaxPool2D 2x2"));
  TEST_ASSERT_NOT_NULL(strstr(text, "Flatten"));
  TEST_ASSERT_NOT_NULL(strstr(text, "Dense 2 + Softmax"));
  TEST_ASSERT_NOT_NULL(strstr(text, "Total parameters: 38"));

  /* snprintf style: a small buffer is cut, the size is still reported. */
  char tiny[16];
  TEST_ASSERT_EQUAL_size_t(need, nm_network_summary(m, tiny, sizeof(tiny)));
  TEST_ASSERT_EQUAL_size_t(15, strlen(tiny));
  TEST_ASSERT_EQUAL_size_t(need, nm_network_summary(m, NULL, 0));
  nm_network_free(m);
  nm_spec_free(spec);

  spec = classifier_spec();
  m = build_seeded(spec, 1);
  TEST_ASSERT_EQUAL_size_t(2 * 8 + 8 + 8 * 2 + 2, nm_network_parameter_count(m));
  nm_network_summary(m, text, sizeof(text));
  TEST_ASSERT_NOT_NULL(strstr(text, "Dense 8 + Tanh"));
  nm_network_free(m);
  nm_spec_free(spec);
}

typedef struct {
  NmNetwork *net;
  float *x;
  float *expected;
  int ok;
} Worker;

static void *predict_many(void *arg) {
  Worker *w = arg;
  for (int i = 0; i < 300; ++i) {
    NmTensor out;
    if (nm_network_predict(w->net, nm_tensor_2d(w->x, 2, 2), &out) != NM_STATUS_OK) {
      w->ok = 0;
      return NULL;
    }
    if (memcmp(out.data, w->expected, 4 * sizeof(float)) != 0)
      w->ok = 0;
    nm_tensor_free(&out);
  }
  return NULL;
}

void test_one_network_can_be_shared_between_threads(void) {
  float x[4] = {0.5f, -0.5f, 1.0f, 2.0f};
  NmSpec *spec = classifier_spec();
  NmNetwork *m = build_seeded(spec, 8);
  NmTensor ref;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_network_predict(m, nm_tensor_2d(x, 2, 2), &ref));
  Worker w[4];
  pthread_t t[4];
  for (int i = 0; i < 4; ++i) {
    w[i] = (Worker){.net = m, .x = x, .expected = ref.data, .ok = 1};
    pthread_create(&t[i], NULL, predict_many, &w[i]);
  }
  for (int i = 0; i < 4; ++i) {
    pthread_join(t[i], NULL);
    TEST_ASSERT_TRUE(w[i].ok);
  }
  nm_tensor_free(&ref);
  nm_network_free(m);
  nm_spec_free(spec);
}

void test_import_graphir_with_external_weights(void) {
  NmSpec *spec = classifier_spec();
  NmNetwork *m = build_seeded(spec, 4);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_network_save(m, "tmp_nm_api_import"));
  NmNetwork *loaded = nm_network_import_graphir("tmp_nm_api_import/model.graphir.json");
  TEST_ASSERT_NOT_NULL_MESSAGE(loaded, nm_last_error());
  TEST_ASSERT_EQUAL_size_t(nm_network_parameter_count(m),
                           nm_network_parameter_count(loaded));
  TEST_ASSERT_NULL(nm_network_import_graphir("tmp_nm_api_import/missing.json"));
  nm_network_free(loaded);
  nm_network_free(m);
  nm_spec_free(spec);
  remove("tmp_nm_api_import/model.graphir.json");
  remove("tmp_nm_api_import/weights.girw");
  remove("tmp_nm_api_import");
}

#define CORPUS "app/nm_lib/tests/fixtures/conformance/"

/* A bfloat16 model runs promoted in f32 but keeps its type on the way out. */
void test_bfloat16_model_keeps_its_type_and_cannot_be_trained(void) {
  NmNetwork *m = nm_network_import_graphir(CORPUS "dense_relu_softmax_bf16.graphir.json");
  TEST_ASSERT_NOT_NULL_MESSAGE(m, nm_last_error());

  float x[8] = {0.5f, -0.5f, 1.0f, 0.25f, 0.0f, 0.75f, -1.0f, 0.5f};
  float y[6] = {0};
  NmFitOptions f = nm_fit_options_default();
  f.epochs = 1;
  TEST_ASSERT_EQUAL(NM_STATUS_UNSUPPORTED_TYPE,
                    nm_network_fit(m, nm_tensor_2d(x, 2, 4), T2(y, 2, 3), &f, NULL, NULL));

  /* Inline JSON cannot hold bfloat16, so only the package form works. */
  TEST_ASSERT_NOT_EQUAL(NM_STATUS_OK, nm_network_export_graphir(m, "tmp_nm_bf16.json"));

  TEST_ASSERT_EQUAL_MESSAGE(NM_STATUS_OK, nm_network_save(m, "tmp_nm_bf16_model"),
                            nm_last_error());
  char *json = read_file("tmp_nm_bf16_model/model.graphir.json");
  TEST_ASSERT_NOT_NULL(strstr(json, "\"dtype\":\"bfloat16\""));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"storage_dtype\":\"bfloat16\""));
  TEST_ASSERT_NULL(strstr(json, "\"dtype\":\"float32\""));
  free(json);

  NmNetwork *again = nm_network_load("tmp_nm_bf16_model");
  TEST_ASSERT_NOT_NULL_MESSAGE(again, nm_last_error());
  NmTensor a, b;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_network_predict(m, nm_tensor_2d(x, 2, 4), &a));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_network_predict(again, nm_tensor_2d(x, 2, 4), &b));
  TEST_ASSERT_EQUAL_MEMORY(a.data, b.data, 6 * sizeof(float));
  nm_tensor_free(&a);
  nm_tensor_free(&b);
  nm_network_free(again);
  nm_network_free(m);
  remove("tmp_nm_bf16_model/model.graphir.json");
  remove("tmp_nm_bf16_model/weights.girw");
  remove("tmp_nm_bf16_model");
}

/* An attention block runs on the graph interpreter: it predicts, survives an
 * export round trip with its Transpose, LayerNorm and Reshape attributes, and
 * refuses training. */
void test_attention_block_runs_and_round_trips(void) {
  NmNetwork *m = nm_network_import_graphir(CORPUS "attention_block.graphir.json");
  TEST_ASSERT_NOT_NULL_MESSAGE(m, nm_last_error());

  float x[24];
  for (size_t i = 0; i < 24; ++i)
    x[i] = (float)((int)(i * 5 % 11) - 5) / 5.0f;
  NmTensor input = {.data = x, .ndim = 3, .dims = {2, 3, 4}};
  NmTensor a, b;
  TEST_ASSERT_EQUAL_MESSAGE(NM_STATUS_OK, nm_network_predict(m, input, &a),
                            nm_last_error());
  TEST_ASSERT_EQUAL_size_t(24, nm_tensor_numel(&a));

  NmFitOptions f = nm_fit_options_default();
  f.epochs = 1;
  TEST_ASSERT_EQUAL(NM_STATUS_UNSUPPORTED_TYPE,
                    nm_network_fit(m, input, &input, &f, NULL, NULL));

  TEST_ASSERT_EQUAL_MESSAGE(NM_STATUS_OK,
                            nm_network_export_graphir(m, "tmp_nm_attention.json"),
                            nm_last_error());
  char *json = read_file("tmp_nm_attention.json");
  TEST_ASSERT_NOT_NULL(strstr(json, "\"perm\":[0,2,3,1]"));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"normalized_shape\":[4]"));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"opset_version\":2"));
  free(json);
  NmNetwork *again = nm_network_import_graphir("tmp_nm_attention.json");
  TEST_ASSERT_NOT_NULL_MESSAGE(again, nm_last_error());
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_network_predict(again, input, &b));
  TEST_ASSERT_EQUAL_MEMORY(a.data, b.data, 24 * sizeof(float));

  char summary[2048];
  nm_network_summary(m, summary, sizeof(summary));
  TEST_ASSERT_NOT_NULL(strstr(summary, "LayerNorm"));
  TEST_ASSERT_NOT_NULL(strstr(summary, "Transpose"));

  nm_tensor_free(&a);
  nm_tensor_free(&b);
  nm_network_free(again);
  nm_network_free(m);
  remove("tmp_nm_attention.json");
}

/* A model with a float input and a bool padding mask: inputs are found by
 * name in any order, described by nm_network_input_info, and the old
 * one-tensor predict points to the named API. */
void test_named_inputs_with_a_bool_mask(void) {
  NmNetwork *m = nm_network_import_graphir(CORPUS "where_padding_mask.graphir.json");
  TEST_ASSERT_NOT_NULL_MESSAGE(m, nm_last_error());
  TEST_ASSERT_EQUAL_size_t(2, nm_network_input_count(m));
  NmInput info;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_network_input_info(m, 1, &info));
  TEST_ASSERT_EQUAL_STRING("padding", info.name);
  TEST_ASSERT_EQUAL(NM_DTYPE_BOOL, info.dtype);
  TEST_ASSERT_EQUAL_size_t(2, info.ndim);
  TEST_ASSERT_EQUAL_size_t(2, info.dims[0]);
  TEST_ASSERT_EQUAL_size_t(4, info.dims[1]);
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT, nm_network_input_info(m, 2, &info));

  float x[8] = {-1.0f, 0.0f, 1.0f, -0.2f, 0.8f, -0.4f, 0.6f, -0.6f};
  uint8_t padding[8] = {0, 0, 1, 1, 0, 1, 0, 0};
  NmInput inputs[2] = {
      {.name = "padding", .dtype = NM_DTYPE_BOOL, .data = padding, .ndim = 2, .dims = {2, 4}},
      {.name = "x", .dtype = NM_DTYPE_FLOAT32, .data = x, .ndim = 2, .dims = {2, 4}},
  };
  NmTensor out;
  TEST_ASSERT_EQUAL_MESSAGE(NM_STATUS_OK, nm_network_predict_inputs(m, inputs, 2, &out),
                            nm_last_error());
  const float expected[8] = {-1.0f, 0.0f, -2.0f, -2.0f, 0.8f, -2.0f, 0.6f, -0.6f};
  TEST_ASSERT_EQUAL_size_t(2, out.dims[0]);
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, out.data, 8);
  nm_tensor_free(&out);

  /* Wrong type, wrong shape, unknown name, missing input. */
  inputs[0].dtype = NM_DTYPE_FLOAT32;
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT, nm_network_predict_inputs(m, inputs, 2, &out));
  inputs[0].dtype = NM_DTYPE_BOOL;
  inputs[0].dims[0] = 3;
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT, nm_network_predict_inputs(m, inputs, 2, &out));
  inputs[0].dims[0] = 2;
  inputs[0].name = "mask";
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT, nm_network_predict_inputs(m, inputs, 2, &out));
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT, nm_network_predict_inputs(m, inputs + 1, 1, &out));
  TEST_ASSERT_NOT_NULL(strstr(nm_last_error(), "nm_network_predict_inputs"));

  NmTensor single = {.data = x, .ndim = 2, .dims = {2, 4}};
  TEST_ASSERT_NOT_EQUAL(NM_STATUS_OK, nm_network_predict(m, single, &out));
  TEST_ASSERT_NOT_NULL(strstr(nm_last_error(), "nm_network_predict_inputs"));
  nm_network_free(m);
}

/* int64 token ids through the named API with a NULL name (by position); a
 * single-input model also keeps the plain float API for float graphs. */
void test_token_ids_by_position(void) {
  NmNetwork *m = nm_network_import_graphir(CORPUS "ids_padding_embedding.graphir.json");
  TEST_ASSERT_NOT_NULL_MESSAGE(m, nm_last_error());
  NmInput info;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_network_input_info(m, 0, &info));
  TEST_ASSERT_EQUAL(NM_DTYPE_INT64, info.dtype);
  TEST_ASSERT_EQUAL_size_t(0, info.dims[0]); /* any batch */

  int64_t ids[4] = {0, 3, 0, 9}; /* one sequence; 9 is out of range */
  NmInput input = {.dtype = NM_DTYPE_INT64, .data = ids, .ndim = 2, .dims = {1, 4}};
  NmTensor out;
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT, nm_network_predict_inputs(m, &input, 1, &out));
  ids[3] = -1; /* counts from the end */
  TEST_ASSERT_EQUAL_MESSAGE(NM_STATUS_OK, nm_network_predict_inputs(m, &input, 1, &out),
                            nm_last_error());
  TEST_ASSERT_EQUAL_size_t(1, out.dims[0]);
  TEST_ASSERT_EQUAL_size_t(12, out.dims[1]);
  TEST_ASSERT_EQUAL_FLOAT(9.0f, out.data[0]); /* padding token */
  TEST_ASSERT_EQUAL_FLOAT(9.0f, out.data[8]);
  TEST_ASSERT_NOT_EQUAL(9.0f, out.data[3]);
  nm_tensor_free(&out);
  nm_network_free(m);
}

/* Graphs with token inputs, masks and the tensor operations export to
 * GraphIR and come back with the same predictions. */
void test_token_and_mask_graphs_round_trip_through_export(void) {
  const char *cases[] = {"where_padding_mask", "ids_padding_embedding", "range_positions",
                         "expand_slice_squeeze", "any_row_reduce_max", "cast_float_int64",
                         "equal_not_cast", "gather_embedding"};
  for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
    char path[256];
    snprintf(path, sizeof(path), CORPUS "%s.graphir.json", cases[c]);
    NmNetwork *m = nm_network_import_graphir(path);
    TEST_ASSERT_NOT_NULL_MESSAGE(m, nm_last_error());
    TEST_ASSERT_EQUAL_MESSAGE(NM_STATUS_OK, nm_network_export_graphir(m, "tmp_nm_typed.json"),
                              cases[c]);
    NmNetwork *again = nm_network_import_graphir("tmp_nm_typed.json");
    TEST_ASSERT_NOT_NULL_MESSAGE(again, nm_last_error());
    TEST_ASSERT_EQUAL_size_t(nm_network_input_count(m), nm_network_input_count(again));

    /* Inputs as declared, batch 2 where the batch is open. */
    NmInput inputs[2];
    float floats[2][64];
    int64_t ints[2][64];
    uint8_t bools[2][64];
    for (size_t i = 0; i < nm_network_input_count(m); ++i) {
      TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_network_input_info(m, i, &inputs[i]));
      if (inputs[i].dims[0] == 0)
        inputs[i].dims[0] = 2;
      size_t n = 1;
      for (size_t d = 0; d < inputs[i].ndim; ++d)
        n *= inputs[i].dims[d];
      TEST_ASSERT_TRUE(n <= 64);
      for (size_t k = 0; k < n; ++k) {
        floats[i][k] = (float)((int)(k * 7 % 11) - 5) / 5.0f;
        ints[i][k] = (int64_t)(k % 3);
        bools[i][k] = (uint8_t)(k % 3 == 1);
      }
      inputs[i].data = inputs[i].dtype == NM_DTYPE_INT64  ? (const void *)ints[i]
                       : inputs[i].dtype == NM_DTYPE_BOOL ? (const void *)bools[i]
                                                          : (const void *)floats[i];
    }
    NmTensor a, b;
    size_t count = nm_network_input_count(m);
    TEST_ASSERT_EQUAL_MESSAGE(NM_STATUS_OK, nm_network_predict_inputs(m, inputs, count, &a),
                              cases[c]);
    TEST_ASSERT_EQUAL_MESSAGE(NM_STATUS_OK, nm_network_predict_inputs(again, inputs, count, &b),
                              cases[c]);
    TEST_ASSERT_EQUAL_size_t(nm_tensor_numel(&a), nm_tensor_numel(&b));
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(a.data, b.data, nm_tensor_numel(&a) * sizeof(float),
                                     cases[c]);
    nm_tensor_free(&a);
    nm_tensor_free(&b);
    nm_network_free(again);
    nm_network_free(m);
  }
  remove("tmp_nm_typed.json");
}

/* A model with a declared batch of 2 takes 5 examples in chunks; each row
 * equals the prediction of its example run on its own (padded) chunk. */
void test_fixed_batch_models_take_any_number_of_examples(void) {
  NmNetwork *m = nm_network_import_graphir(CORPUS "where_padding_mask.graphir.json");
  TEST_ASSERT_NOT_NULL_MESSAGE(m, nm_last_error());
  float x[20];
  uint8_t padding[20];
  for (size_t i = 0; i < 20; ++i) {
    x[i] = (float)i * 0.25f - 2.0f;
    padding[i] = (uint8_t)(i % 3 == 0);
  }
  NmInput inputs[2] = {
      {.name = "padding", .dtype = NM_DTYPE_BOOL, .data = padding, .ndim = 2, .dims = {5, 4}},
      {.name = "x", .dtype = NM_DTYPE_FLOAT32, .data = x, .ndim = 2, .dims = {5, 4}},
  };
  NmTensor out;
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT, nm_network_predict_inputs(m, inputs, 2, &out));
  TEST_ASSERT_EQUAL_MESSAGE(NM_STATUS_OK, nm_network_predict_batches(m, inputs, 2, &out),
                            nm_last_error());
  TEST_ASSERT_EQUAL_size_t(5, out.dims[0]);
  TEST_ASSERT_EQUAL_size_t(4, out.dims[1]);
  for (size_t i = 0; i < 20; ++i)
    TEST_ASSERT_EQUAL_FLOAT(padding[i] ? -2.0f : x[i], out.data[i]);
  nm_tensor_free(&out);

  /* Two examples: the same as the plain call. */
  inputs[0].dims[0] = inputs[1].dims[0] = 2;
  NmTensor a, b;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_network_predict_batches(m, inputs, 2, &a));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_network_predict_inputs(m, inputs, 2, &b));
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(b.data, a.data, 8);
  nm_tensor_free(&a);
  nm_tensor_free(&b);

  /* Inputs must agree on the number of examples. */
  inputs[0].dims[0] = 3;
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT, nm_network_predict_batches(m, inputs, 2, &out));
  nm_network_free(m);
}
