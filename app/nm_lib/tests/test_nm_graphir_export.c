/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm_engine_c.h"
#include "nm_export_graphir.h"
#include "nm_import_graphir.h"
#include "nm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNITY_INCLUDE_FLOAT
#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static char *read_text_file(const char *path) {
  FILE *file = fopen(path, "rb");
  TEST_ASSERT_NOT_NULL(file);
  TEST_ASSERT_EQUAL_INT(0, fseek(file, 0, SEEK_END));
  long size = ftell(file);
  TEST_ASSERT_TRUE(size >= 0);
  TEST_ASSERT_EQUAL_INT(0, fseek(file, 0, SEEK_SET));
  char *text = calloc((size_t)size + 1, 1);
  TEST_ASSERT_NOT_NULL(text);
  TEST_ASSERT_EQUAL_size_t((size_t)size,
                           fread(text, 1, (size_t)size, file));
  TEST_ASSERT_EQUAL_INT(0, fclose(file));
  return text;
}

static NmFrontendModel *dense_model(void) {
  NmFrontendModel *model =
      nm_frontend_create((NmShape){.ndim = 1, .dims = {3}});
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_frontend_add_dense(
                        model, &(NmDenseConfig){.units = 4,
                                                 .activation = NM_ACT_RELU}));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_frontend_add_dense(
                        model, &(NmDenseConfig){.units = 2,
                                                 .activation = NM_ACT_SOFTMAX}));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_frontend_finalize(model));
  return model;
}

void test_graphir_export_writes_v4_dense_program(void) {
  const char *path = "tmp_nm_graphir_export.json";
  NmFrontendModel *model = dense_model();
  NmGraphIrExportOptions options = {.batch_size = 2};

  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_export_graphir_json(model, &options, path));
  char *json = read_text_file(path);
  TEST_ASSERT_NOT_NULL(strstr(json, "\"format_version\":4"));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"operation\":\"MatMul\""));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"operation\":\"Add\""));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"operation\":\"ReLU\""));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"operation\":\"Softmax\""));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"shape\":[2,3]"));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"opset_version\":1"));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"kind\":\"inline\""));

  free(json);
  remove(path);
  nm_frontend_free(model);
}

void test_graphir_export_writes_cnn_program_with_opset2(void) {
  const char *path = "tmp_nm_graphir_export_cnn.json";
  NmFrontendModel *model =
      nm_frontend_create((NmShape){.ndim = 3, .dims = {1, 6, 6}});
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(
      NM_STATUS_OK,
      nm_frontend_add_conv2d(model, &(NmConv2dConfig){.filters = 2,
                                                      .kernel_size = 3,
                                                      .stride = 1,
                                                      .padding = 1,
                                                      .activation = NM_ACT_RELU}));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_frontend_add_maxpool2d(
                        model, &(NmPool2dConfig){.kernel_size = 2, .stride = 2}));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_frontend_add_flatten(model));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_frontend_add_dense(
                        model, &(NmDenseConfig){.units = 3,
                                                 .activation = NM_ACT_SOFTMAX}));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_frontend_finalize(model));

  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_export_graphir_json(
                        model, &(NmGraphIrExportOptions){.batch_size = 4},
                        path));
  char *json = read_text_file(path);
  TEST_ASSERT_NOT_NULL(strstr(json, "\"format_version\":4"));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"opset_version\":2"));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"shape\":[4,1,6,6]"));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"shape\":[2,1,3,3]"));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"operation\":\"Conv2D\""));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"strides\":[1,1],\"pads\":[1,1,1,1],"
                                    "\"dilations\":[1,1],\"groups\":1"));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"operation\":\"MaxPool2D\""));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"kernel\":[2,2],\"strides\":[2,2],"
                                    "\"pads\":[0,0,0,0]"));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"operation\":\"Reshape\""));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"shape\":[4,18]"));
  TEST_ASSERT_NOT_NULL(strstr(json, "\"operation\":\"Softmax\""));

  free(json);
  remove(path);
  nm_frontend_free(model);
}

void test_graphir_export_rejects_unfinalized_model_and_zero_batch(void) {
  NmFrontendModel *model =
      nm_frontend_create((NmShape){.ndim = 1, .dims = {3}});
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_frontend_add_dense(
                        model, &(NmDenseConfig){.units = 2,
                                                 .activation = NM_ACT_NONE}));
  NmGraphIrExportOptions options = {.batch_size = 1};
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT,
                    nm_export_graphir_json(model, &options, "ignored.json"));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_frontend_finalize(model));
  options.batch_size = 0;
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT,
                    nm_export_graphir_json(model, &options, "ignored.json"));
  nm_frontend_free(model);
}

static FloatMatrix *ramp_input(size_t rows, size_t cols) {
  FloatMatrix *input = sm_create(rows, cols);
  TEST_ASSERT_NOT_NULL(input);
  for (size_t i = 0; i < rows * cols; ++i)
    input->values[i] = (float)((int)(i * 7U % 13U) - 6) / 6.0f;
  return input;
}

/* Exports `model`, loads the file back through the GraphIR import path and
 * checks that both engines predict the same values. */
static void assert_graphir_roundtrip_matches(NmFrontendModel *model,
                                             size_t input_cols) {
  const char *path = "tmp_nm_graphir_roundtrip.json";
  const size_t batch = 3;
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_export_graphir_json(
                        model, &(NmGraphIrExportOptions){.batch_size = batch},
                        path));

  NmEngineC *direct = nm_engine_c_create(NULL);
  NmEngineC *imported = nm_engine_c_create(NULL);
  NmGraphModel *loaded = NULL;
  TEST_ASSERT_NOT_NULL(direct);
  TEST_ASSERT_NOT_NULL(imported);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare(direct, model));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_prepare_graphir_json(imported, path, &loaded));
  TEST_ASSERT_NOT_NULL(loaded);

  FloatMatrix *input = ramp_input(batch, input_cols);
  FloatMatrix *expected = NULL;
  FloatMatrix *actual = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_predict(direct, model, input, &expected));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_predict_graph(imported, loaded, input, &actual));
  TEST_ASSERT_EQUAL_size_t(expected->rows, actual->rows);
  TEST_ASSERT_EQUAL_size_t(expected->cols, actual->cols);
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected->values, actual->values,
                                expected->rows * expected->cols);

  sm_destroy(actual);
  sm_destroy(expected);
  sm_destroy(input);
  nm_engine_c_free(imported);
  nm_engine_c_free(direct);
  nm_graph_model_free(loaded);
  remove(path);
}

void test_graphir_import_roundtrips_dense_model(void) {
  NmFrontendModel *model = dense_model();
  assert_graphir_roundtrip_matches(model, 3);
  nm_frontend_free(model);
}

void test_graphir_import_roundtrips_cnn_model(void) {
  NmFrontendModel *model =
      nm_frontend_create((NmShape){.ndim = 3, .dims = {1, 6, 6}});
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(
      NM_STATUS_OK,
      nm_frontend_add_conv2d(model, &(NmConv2dConfig){.filters = 2,
                                                      .kernel_size = 3,
                                                      .stride = 1,
                                                      .padding = 1,
                                                      .activation = NM_ACT_RELU}));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_frontend_add_maxpool2d(
                        model, &(NmPool2dConfig){.kernel_size = 2, .stride = 2}));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_frontend_add_flatten(model));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_frontend_add_dense(
                        model, &(NmDenseConfig){.units = 3,
                                                 .activation = NM_ACT_SOFTMAX}));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_frontend_finalize(model));
  assert_graphir_roundtrip_matches(model, 36);
  nm_frontend_free(model);
}

void test_graphir_import_rejects_missing_and_malformed_files(void) {
  NmGraphModel *model = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_IO_ERROR,
                    nm_import_graphir_json("does_not_exist.json", &model));
  TEST_ASSERT_NULL(model);

  const char *path = "tmp_nm_graphir_bad.json";
  FILE *file = fopen(path, "wb");
  TEST_ASSERT_NOT_NULL(file);
  fputs("{\"format_version\":4,\"values\":[", file);
  fclose(file);
  TEST_ASSERT_EQUAL(NM_STATUS_FORMAT_ERROR,
                    nm_import_graphir_json(path, &model));
  TEST_ASSERT_NULL(model);
  remove(path);
}

void test_graph_prepared_engine_rejects_model_fit_and_foreign_graphs(void) {
  const char *path = "tmp_nm_graphir_graph_only.json";
  NmFrontendModel *model = dense_model();
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_export_graphir_json(
                        model, &(NmGraphIrExportOptions){.batch_size = 1},
                        path));
  NmEngineC *engine = nm_engine_c_create(NULL);
  NmGraphModel *graph = NULL;
  NmGraphModel *other = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_prepare_graphir_json(engine, path, &graph));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_import_graphir_json(path, &other));

  FloatMatrix *input = ramp_input(2, 3);
  FloatMatrix *targets = sm_create(2, 2);
  FloatMatrix *prediction = NULL;
  TEST_ASSERT_NOT_NULL(targets);
  NmTrainConfig train = {.batch_size = 1, .learning_rate = 0.1f, .epochs = 1};
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT,
                    nm_engine_c_fit(engine, model, input, targets, &train));
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT,
                    nm_engine_c_predict(engine, model, input, &prediction));
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT,
                    nm_engine_c_predict_graph(engine, other, input, &prediction));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_predict_graph(engine, graph, input, &prediction));
  TEST_ASSERT_EQUAL_size_t(2, prediction->rows);

  sm_destroy(prediction);
  sm_destroy(targets);
  sm_destroy(input);
  nm_graph_model_free(other);
  nm_graph_model_free(graph);
  nm_engine_c_free(engine);
  nm_frontend_free(model);
  remove(path);
}

void test_engine_rejects_graphs_with_unsupported_operations(void) {
  const char *path = "tmp_nm_graphir_unsupported.json";
  FILE *file = fopen(path, "wb");
  TEST_ASSERT_NOT_NULL(file);
  fputs("{\"format_version\":4,\"opset_version\":2,\"name\":\"g\","
        "\"metadata\":{},\"inputs\":[0],\"outputs\":[1],\"values\":["
        "{\"id\":0,\"name\":\"x\",\"producer\":0,\"tensor\":{\"dtype\":"
        "\"float32\",\"shape\":[1,2],\"layout\":null,\"quantization\":null},"
        "\"data\":null},{\"id\":1,\"name\":\"y\",\"producer\":1,\"tensor\":"
        "{\"dtype\":\"float32\",\"shape\":[1,2],\"layout\":null,"
        "\"quantization\":null},\"data\":null}],\"nodes\":[{\"id\":0,"
        "\"name\":\"x\",\"operation\":\"Input\",\"inputs\":[],\"outputs\":"
        "[0],\"attributes\":{},\"metadata\":{}},{\"id\":1,\"name\":\"y\","
        "\"operation\":\"Neg\",\"inputs\":[0],\"outputs\":[1],"
        "\"attributes\":{},\"metadata\":{}}]}",
        file);
  fclose(file);

  NmGraphModel *graph = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_UNSUPPORTED_TYPE,
                    nm_import_graphir_json(path, &graph));
  TEST_ASSERT_NULL(graph);
  remove(path);
}

/* `x[2,8] * W[8,3] + b[3]` with W and b read from the GraphIR reference
 * package (tests/fixtures/weights_v2.girw, written by GraphIR itself). */
static const char *EXTERNAL_MODEL =
    "{\"format_version\":4,\"opset_version\":1,\"name\":\"external\","
    "\"metadata\":{},\"inputs\":[0],\"outputs\":[4],\"values\":["
    "{\"id\":0,\"name\":\"x\",\"producer\":0,\"tensor\":{\"dtype\":\"float32\","
    "\"shape\":[2,8],\"layout\":null,\"quantization\":null},\"data\":null},"
    "{\"id\":1,\"name\":\"w\",\"producer\":1,\"tensor\":{\"dtype\":\"float32\","
    "\"shape\":[8,3],\"layout\":null,\"quantization\":null},\"data\":{"
    "\"kind\":\"external\",\"package\":\"app/nm_lib/tests/fixtures/weights_v2.girw\","
    "\"offset\":192,\"byte_length\":96,\"element_count\":24,"
    "\"storage_dtype\":\"float32\"}},"
    "{\"id\":2,\"name\":\"b\",\"producer\":2,\"tensor\":{\"dtype\":\"float32\","
    "\"shape\":[3],\"layout\":null,\"quantization\":null},\"data\":{"
    "\"kind\":\"external\",\"package\":\"app/nm_lib/tests/fixtures/weights_v2.girw\","
    "\"offset\":320,\"byte_length\":12,\"element_count\":3,"
    "\"storage_dtype\":\"float32\"}},"
    "{\"id\":3,\"name\":\"l\",\"producer\":3,\"tensor\":{\"dtype\":\"float32\","
    "\"shape\":[2,3],\"layout\":null,\"quantization\":null},\"data\":null},"
    "{\"id\":4,\"name\":\"y\",\"producer\":4,\"tensor\":{\"dtype\":\"float32\","
    "\"shape\":[2,3],\"layout\":null,\"quantization\":null},\"data\":null}],"
    "\"nodes\":["
    "{\"id\":0,\"name\":\"x\",\"operation\":\"Input\",\"inputs\":[],\"outputs\":[0],"
    "\"attributes\":{},\"metadata\":{}},"
    "{\"id\":1,\"name\":\"w\",\"operation\":\"Constant\",\"inputs\":[],\"outputs\":[1],"
    "\"attributes\":{},\"metadata\":{}},"
    "{\"id\":2,\"name\":\"b\",\"operation\":\"Constant\",\"inputs\":[],\"outputs\":[2],"
    "\"attributes\":{},\"metadata\":{}},"
    "{\"id\":3,\"name\":\"l\",\"operation\":\"MatMul\",\"inputs\":[0,1],\"outputs\":[3],"
    "\"attributes\":{},\"metadata\":{}},"
    "{\"id\":4,\"name\":\"y\",\"operation\":\"Add\",\"inputs\":[3,2],\"outputs\":[4],"
    "\"attributes\":{},\"metadata\":{}}]}";

void test_graphir_import_reads_weights_from_girw_package(void) {
  const char *path = "tmp_nm_graphir_external.json";
  FILE *file = fopen(path, "wb");
  TEST_ASSERT_NOT_NULL(file);
  fputs(EXTERNAL_MODEL, file);
  fclose(file);

  NmEngineC *engine = nm_engine_c_create(NULL);
  NmGraphModel *graph = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_prepare_graphir_json(engine, path, &graph));
  FloatMatrix *input = ramp_input(2, 8);
  FloatMatrix *prediction = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_predict_graph(engine, graph, input, &prediction));

  const float bias[3] = {0.5f, -1.5f, 2.0f};
  for (size_t row = 0; row < 2; ++row) {
    for (size_t col = 0; col < 3; ++col) {
      float expected = bias[col];
      for (size_t k = 0; k < 8; ++k)
        expected += input->values[row * 8 + k] *
                    ((float)(k * 3 + col) * 0.25f - 3.0f);
      TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected, prediction->values[row * 3 + col]);
    }
  }

  sm_destroy(prediction);
  sm_destroy(input);
  nm_graph_model_free(graph);
  nm_engine_c_free(engine);
  remove(path);
}

void test_graphir_import_rejects_missing_or_wrong_external_weights(void) {
  const char *path = "tmp_nm_graphir_external_bad.json";
  const char *needle = "\"offset\":192";
  char *broken = strdup(EXTERNAL_MODEL);
  TEST_ASSERT_NOT_NULL(broken);
  char *hit = strstr(broken, needle);
  TEST_ASSERT_NOT_NULL(hit);
  hit[10] = '3'; /* offset 132 does not match any entry */
  FILE *file = fopen(path, "wb");
  fputs(broken, file);
  fclose(file);
  NmGraphModel *graph = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_FORMAT_ERROR, nm_import_graphir_json(path, &graph));
  TEST_ASSERT_NULL(graph);

  char *missing = strdup(EXTERNAL_MODEL);
  hit = strstr(missing, "weights_v2.girw");
  hit[0] = 'x'; /* package file does not exist */
  file = fopen(path, "wb");
  fputs(missing, file);
  fclose(file);
  TEST_ASSERT_EQUAL(NM_STATUS_IO_ERROR, nm_import_graphir_json(path, &graph));
  TEST_ASSERT_NULL(graph);
  free(broken);
  free(missing);
  remove(path);
}

/* ---- Training through the engine matches the legacy trainer ------------- */

typedef enum { KIND_DENSE, KIND_CNN } ModelKind;

static void build_frontend(NmFrontendModel *m, ModelKind kind) {
  if (kind == KIND_CNN) {
    TEST_ASSERT_EQUAL(NM_STATUS_OK,
                      nm_frontend_add_conv2d(
                          m, &(NmConv2dConfig){.filters = 2, .kernel_size = 3,
                                               .stride = 1, .padding = 1,
                                               .activation = NM_ACT_RELU}));
    TEST_ASSERT_EQUAL(NM_STATUS_OK,
                      nm_frontend_add_maxpool2d(
                          m, &(NmPool2dConfig){.kernel_size = 2, .stride = 2}));
    TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_frontend_add_flatten(m));
  } else {
    TEST_ASSERT_EQUAL(NM_STATUS_OK,
                      nm_frontend_add_dense(
                          m, &(NmDenseConfig){.units = 6,
                                              .activation = NM_ACT_RELU}));
  }
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_frontend_add_dense(
                        m, &(NmDenseConfig){.units = 3,
                                            .activation = NM_ACT_SOFTMAX}));
}

/* The same topology through the legacy API. */
static void build_legacy(NmModel *m, ModelKind kind) {
  if (kind == KIND_CNN) {
    TEST_ASSERT_EQUAL(NM_STATUS_OK,
                      nm_add_conv2d(m, &(NmConv2dConfig){.filters = 2,
                                                         .kernel_size = 3,
                                                         .stride = 1,
                                                         .padding = 1,
                                                         .activation = NM_ACT_RELU}));
    TEST_ASSERT_EQUAL(NM_STATUS_OK,
                      nm_add_maxpool2d(m, &(NmPool2dConfig){.kernel_size = 2,
                                                            .stride = 2}));
    TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_flatten(m));
  } else {
    TEST_ASSERT_EQUAL(NM_STATUS_OK,
                      nm_add_dense(m, &(NmDenseConfig){.units = 6,
                                                       .activation = NM_ACT_RELU}));
  }
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_add_dense(m, &(NmDenseConfig){.units = 3,
                                                     .activation = NM_ACT_SOFTMAX}));
}

static NmShape shape_for(ModelKind kind) {
  return kind == KIND_CNN ? (NmShape){.ndim = 3, .dims = {1, 6, 6}}
                          : (NmShape){.ndim = 1, .dims = {8}};
}

static size_t input_cols_for(ModelKind kind) { return kind == KIND_CNN ? 36 : 8; }

static void make_data(size_t rows, size_t cols, FloatMatrix **x, FloatMatrix **y) {
  *x = ramp_input(rows, cols);
  *y = sm_create(rows, 3);
  TEST_ASSERT_NOT_NULL(*y);
  for (size_t r = 0; r < rows; ++r)
    (*y)->values[r * 3 + (r % 3)] = 1.0f;
}

static NmTrainConfig train_config(void) {
  return (NmTrainConfig){.batch_size = 4, .learning_rate = 0.05f, .epochs = 5};
}

/* Same seed, same data: the engine's graph trainer and nm_fit must agree. */
static void assert_engine_fit_matches_legacy(ModelKind kind) {
  const size_t rows = 12;
  FloatMatrix *x = NULL, *y = NULL;
  make_data(rows, input_cols_for(kind), &x, &y);

  nm_set_seed(1234);
  NmFrontendModel *via_engine = nm_frontend_create(shape_for(kind));
  build_frontend(via_engine, kind);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_frontend_finalize(via_engine));

  nm_set_seed(1234);
  NmModel *legacy = nm_create(shape_for(kind));
  build_legacy(legacy, kind);
  NmCompileConfig compile = nm_compile_defaults();
  compile.loss_type = NM_LOSS_CROSS_ENTROPY;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_compile(legacy, &compile));

  NmTrainConfig config = train_config();
  NmEngineC *engine = nm_engine_c_create(NULL);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare(engine, via_engine));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_fit(engine, via_engine, x, y, &config));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_fit(legacy, x, y, &config));

  FloatMatrix *expected = NULL, *actual = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_predict(legacy, x, &expected));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_predict(engine, via_engine, x, &actual));
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected->values, actual->values, rows * 3);

  sm_destroy(actual);
  sm_destroy(expected);
  nm_engine_c_free(engine);
  nm_free(legacy);
  nm_frontend_free(via_engine);
  sm_destroy(y);
  sm_destroy(x);
}

void test_engine_fit_matches_legacy_fit_for_dense_model(void) {
  assert_engine_fit_matches_legacy(KIND_DENSE);
}

void test_engine_fit_matches_legacy_fit_for_cnn_model(void) {
  assert_engine_fit_matches_legacy(KIND_CNN);
}

/* An imported graph owns its weights and trains in place; starting from the
 * same weights it must reach the same result as the model it was exported from. */
void test_imported_graph_trains_like_the_exported_model(void) {
  const char *path = "tmp_nm_graphir_train.json";
  const size_t rows = 12;
  FloatMatrix *x = NULL, *y = NULL;
  make_data(rows, input_cols_for(KIND_CNN), &x, &y);

  nm_set_seed(77);
  NmFrontendModel *model = nm_frontend_create(shape_for(KIND_CNN));
  build_frontend(model, KIND_CNN);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_frontend_finalize(model));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_export_graphir_json(
                        model, &(NmGraphIrExportOptions){.batch_size = 1}, path));

  NmEngineC *direct = nm_engine_c_create(NULL);
  NmEngineC *imported = nm_engine_c_create(NULL);
  NmGraphModel *graph = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare(direct, model));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_prepare_graphir_json(imported, path, &graph));

  NmTrainConfig config = train_config();
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_fit(direct, model, x, y, &config));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_fit_graph(imported, graph, x, y, &config));

  FloatMatrix *expected = NULL, *actual = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_predict(direct, model, x, &expected));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_predict_graph(imported, graph, x, &actual));
  /* Export prints weights with 9 digits, so the start differs by one ulp. */
  for (size_t i = 0; i < rows * 3; ++i)
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected->values[i], actual->values[i]);

  sm_destroy(actual);
  sm_destroy(expected);
  nm_graph_model_free(graph);
  nm_engine_c_free(imported);
  nm_engine_c_free(direct);
  nm_frontend_free(model);
  sm_destroy(y);
  sm_destroy(x);
  remove(path);
}

/* ---- The frontend owns a graph, not a sequential model ------------------- */

void test_frontend_is_closed_after_finalize_and_exposes_the_graph(void) {
  NmFrontendModel *model = nm_frontend_create((NmShape){.ndim = 1, .dims = {4}});
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_NULL(nm_frontend_graph(model));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_frontend_add_dense(model, &(NmDenseConfig){
                                                     .units = 2,
                                                     .activation = NM_ACT_NONE}));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_frontend_finalize(model));
  TEST_ASSERT_NOT_NULL(nm_frontend_graph(model));

  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT,
                    nm_frontend_add_dense(model, &(NmDenseConfig){
                                                     .units = 2,
                                                     .activation = NM_ACT_NONE}));
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT, nm_frontend_finalize(model));
  nm_frontend_free(model);
}

void test_engine_rejects_unfinalized_frontend_model(void) {
  NmFrontendModel *model = nm_frontend_create((NmShape){.ndim = 1, .dims = {4}});
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_frontend_add_dense(model, &(NmDenseConfig){
                                                     .units = 2,
                                                     .activation = NM_ACT_NONE}));
  NmEngineC *engine = nm_engine_c_create(NULL);
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT, nm_engine_c_prepare(engine, model));
  nm_engine_c_free(engine);
  nm_frontend_free(model);
}

/* A model saved with the legacy API loads into a frontend model whose graph
 * owns copies of the weights and predicts the same values. */
static void assert_legacy_file_imports(ModelKind kind) {
  const char *path = "tmp_nm_legacy_model.nms";
  const size_t rows = 5;
  FloatMatrix *x = ramp_input(rows, input_cols_for(kind));

  nm_set_seed(99);
  NmModel *legacy = nm_create(shape_for(kind));
  build_legacy(legacy, kind);
  NmCompileConfig compile = nm_compile_defaults();
  compile.loss_type = NM_LOSS_CROSS_ENTROPY;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_compile(legacy, &compile));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_save(legacy, path));

  NmFrontendModel *imported = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_frontend_import_legacy(path, &imported));
  NmEngineC *engine = nm_engine_c_create(NULL);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare(engine, imported));

  FloatMatrix *expected = NULL, *actual = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_predict(legacy, x, &expected));
  /* The legacy model can go away: the graph holds its own weights. */
  nm_free(legacy);
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_predict(engine, imported, x, &actual));
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected->values, actual->values, rows * 3);

  sm_destroy(actual);
  sm_destroy(expected);
  nm_engine_c_free(engine);
  nm_frontend_free(imported);
  sm_destroy(x);
  remove(path);
}

void test_legacy_dense_file_imports_into_the_frontend(void) {
  assert_legacy_file_imports(KIND_DENSE);
}

void test_legacy_cnn_file_imports_into_the_frontend(void) {
  assert_legacy_file_imports(KIND_CNN);
}

/* Weights trained on an imported graph can be exported again. */
void test_trained_graph_exports_and_reimports_with_its_new_weights(void) {
  const char *first = "tmp_nm_graphir_first.json";
  const char *second = "tmp_nm_graphir_second.json";
  FloatMatrix *x = NULL, *y = NULL;
  make_data(12, input_cols_for(KIND_DENSE), &x, &y);

  nm_set_seed(3);
  NmFrontendModel *model = nm_frontend_create(shape_for(KIND_DENSE));
  build_frontend(model, KIND_DENSE);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_frontend_finalize(model));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_export_graphir_json(
                        model, &(NmGraphIrExportOptions){.batch_size = 1}, first));

  NmEngineC *trainer = nm_engine_c_create(NULL);
  NmGraphModel *graph = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_prepare_graphir_json(trainer, first, &graph));
  NmTrainConfig config = train_config();
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_fit_graph(trainer, graph, x, y, &config));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_export_graphir_graph_json(
                        graph, &(NmGraphIrExportOptions){.batch_size = 1}, second));

  NmEngineC *reader = nm_engine_c_create(NULL);
  NmGraphModel *reloaded = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_prepare_graphir_json(reader, second, &reloaded));
  FloatMatrix *expected = NULL, *actual = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_predict_graph(trainer, graph, x, &expected));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_predict_graph(reader, reloaded, x, &actual));
  for (size_t i = 0; i < 12 * 3; ++i)
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, expected->values[i], actual->values[i]);

  sm_destroy(actual);
  sm_destroy(expected);
  nm_graph_model_free(reloaded);
  nm_engine_c_free(reader);
  nm_graph_model_free(graph);
  nm_engine_c_free(trainer);
  nm_frontend_free(model);
  sm_destroy(y);
  sm_destroy(x);
  remove(first);
  remove(second);
}

void test_engine_requires_prepare_and_predicts_finalized_model(void) {
  NmFrontendModel *model = dense_model();
  NmEngineC *engine = nm_engine_c_create(NULL);
  TEST_ASSERT_NOT_NULL(engine);
  FloatMatrix *input = sm_create(2, 3);
  TEST_ASSERT_NOT_NULL(input);
  FloatMatrix *prediction = NULL;

  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT,
                    nm_engine_c_predict(engine, model, input, &prediction));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare(engine, model));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_predict(engine, model, input, &prediction));
  TEST_ASSERT_NOT_NULL(prediction);
  TEST_ASSERT_EQUAL_size_t(2, prediction->rows);
  TEST_ASSERT_EQUAL_size_t(2, prediction->cols);

  sm_destroy(prediction);
  sm_destroy(input);
  nm_engine_c_free(engine);
  nm_frontend_free(model);
}

void test_engine_capabilities_are_safe_across_model_lifetimes(void) {
  const NmBackendCapabilities *caps_null = nm_engine_c_capabilities(NULL);
  TEST_ASSERT_NOT_NULL(caps_null);

  NmEngineC *engine = nm_engine_c_create(NULL);
  TEST_ASSERT_NOT_NULL(engine);

  const NmBackendCapabilities *caps_unprepared = nm_engine_c_capabilities(engine);
  TEST_ASSERT_NOT_NULL(caps_unprepared);
  TEST_ASSERT_EQUAL_STRING(caps_null->name, caps_unprepared->name);

  NmFrontendModel *model = dense_model();
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare(engine, model));

  const NmBackendCapabilities *caps_prepared = nm_engine_c_capabilities(engine);
  TEST_ASSERT_NOT_NULL(caps_prepared);

  /* Free model before engine and verify querying capabilities remains safe (no UAF) */
  nm_frontend_free(model);
  const NmBackendCapabilities *caps_after_free = nm_engine_c_capabilities(engine);
  TEST_ASSERT_NOT_NULL(caps_after_free);
  TEST_ASSERT_EQUAL_STRING(caps_prepared->name, caps_after_free->name);

  nm_engine_c_free(engine);
}

void test_engine_rejects_unprepared_or_mismatched_model(void) {
  NmFrontendModel *model_a = dense_model();
  NmFrontendModel *model_b = dense_model();
  NmEngineC *engine = nm_engine_c_create(NULL);
  TEST_ASSERT_NOT_NULL(engine);

  FloatMatrix *input = sm_create(2, 3);
  TEST_ASSERT_NOT_NULL(input);
  FloatMatrix *prediction = NULL;

  /* Reject before prepare */
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT,
                    nm_engine_c_predict(engine, model_a, input, &prediction));

  /* Prepare on model_a */
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare(engine, model_a));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_predict(engine, model_a, input, &prediction));
  sm_destroy(prediction);
  prediction = NULL;

  /* Reject on model_b (engine is prepared for model_a, not model_b) */
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT,
                    nm_engine_c_predict(engine, model_b, input, &prediction));

  sm_destroy(input);
  nm_engine_c_free(engine);
  nm_frontend_free(model_a);
  nm_frontend_free(model_b);
}

void test_multiple_engines_can_share_finalized_model(void) {
  NmFrontendModel *model = dense_model();
  NmEngineC *engine1 = nm_engine_c_create(NULL);
  NmEngineC *engine2 = nm_engine_c_create(NULL);
  TEST_ASSERT_NOT_NULL(engine1);
  TEST_ASSERT_NOT_NULL(engine2);

  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare(engine1, model));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare(engine2, model));

  FloatMatrix *input_batch2 = sm_create(2, 3);
  FloatMatrix *input_batch4 = sm_create(4, 3);
  TEST_ASSERT_NOT_NULL(input_batch2);
  TEST_ASSERT_NOT_NULL(input_batch4);

  FloatMatrix *pred1 = NULL;
  FloatMatrix *pred2 = NULL;

  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_predict(engine1, model, input_batch2, &pred1));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_engine_c_predict(engine2, model, input_batch4, &pred2));

  TEST_ASSERT_NOT_NULL(pred1);
  TEST_ASSERT_NOT_NULL(pred2);
  TEST_ASSERT_EQUAL_size_t(2, pred1->rows);
  TEST_ASSERT_EQUAL_size_t(2, pred1->cols);
  TEST_ASSERT_EQUAL_size_t(4, pred2->rows);
  TEST_ASSERT_EQUAL_size_t(2, pred2->cols);

  sm_destroy(pred1);
  sm_destroy(pred2);
  sm_destroy(input_batch2);
  sm_destroy(input_batch4);
  nm_engine_c_free(engine1);
  nm_engine_c_free(engine2);
  nm_frontend_free(model);
}

void test_engine_rejects_unsupported_backend(void) {
  NmFrontendModel *model = dense_model();
  NmEngineCConfig config = nm_engine_c_defaults();
  config.backend = (NmBackendType)99;
  NmEngineC *engine = nm_engine_c_create(&config);
  TEST_ASSERT_NOT_NULL(engine);

  TEST_ASSERT_EQUAL(NM_STATUS_UNSUPPORTED_TYPE,
                    nm_engine_c_prepare(engine, model));

  FloatMatrix *input = sm_create(1, 3);
  FloatMatrix *pred = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT,
                    nm_engine_c_predict(engine, model, input, &pred));
  TEST_ASSERT_NULL(pred);

  sm_destroy(input);
  nm_engine_c_free(engine);
  nm_frontend_free(model);
}
