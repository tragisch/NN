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
#include "nm_weightset.h"
#include "nm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNITY_INCLUDE_FLOAT
#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

typedef enum { DENSE, CNN } Kind;

static NmShape shape_for(Kind kind) {
  return kind == CNN ? (NmShape){.ndim = 3, .dims = {1, 6, 6}}
                     : (NmShape){.ndim = 1, .dims = {8}};
}

static size_t cols_for(Kind kind) { return kind == CNN ? 36 : 8; }

static NmFrontendModel *make_model(Kind kind, uint64_t seed) {
  nm_set_seed(seed);
  NmFrontendModel *m = nm_frontend_create(shape_for(kind));
  if (kind == CNN) {
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
                          m, &(NmDenseConfig){.units = 6, .activation = NM_ACT_RELU}));
  }
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_frontend_add_dense(
                        m, &(NmDenseConfig){.units = 3, .activation = NM_ACT_SOFTMAX}));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_frontend_finalize(m));
  return m;
}

static FloatMatrix *ramp(size_t rows, size_t cols) {
  FloatMatrix *x = sm_create(rows, cols);
  TEST_ASSERT_NOT_NULL(x);
  for (size_t i = 0; i < rows * cols; ++i)
    x->values[i] = (float)((int)(i * 7U % 13U) - 6) / 6.0f;
  return x;
}

/* A transport-like buffer: counts how often it is handed back. */
typedef struct {
  void *memory;
  size_t size;
  int released;
} Message;

static void release_message(void *context) {
  Message *m = context;
  m->released++;
}

static Message make_message(const NmGraphModel *graph, uint64_t version) {
  const uint64_t epoch = 1;
  Message m = {0};
  m.size = nm_weightset_message_size(graph);
  TEST_ASSERT_TRUE(m.size > 64);
  m.memory = aligned_alloc(64, (m.size + 63) / 64 * 64);
  TEST_ASSERT_NOT_NULL(m.memory);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_weightset_write(graph, epoch, version, m.memory, m.size));
  return m;
}

static NmGraphModel *export_and_import(NmFrontendModel *model, const char *path) {
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_export_graphir_json(
                        model, &(NmGraphIrExportOptions){.batch_size = 1}, path));
  NmGraphModel *graph = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_import_graphir_json(path, &graph));
  remove(path);
  return graph;
}

void test_structure_id_ignores_weights_and_tells_topologies_apart(void) {
  NmFrontendModel *a = make_model(DENSE, 1);
  NmFrontendModel *b = make_model(DENSE, 2); /* same topology, other weights */
  NmFrontendModel *c = make_model(CNN, 1);
  uint64_t id_a = nm_graph_structure_id(nm_frontend_graph(a));
  TEST_ASSERT_NOT_EQUAL(0, id_a);
  TEST_ASSERT_EQUAL_UINT64(id_a, nm_graph_structure_id(nm_frontend_graph(b)));
  TEST_ASSERT_NOT_EQUAL(id_a, nm_graph_structure_id(nm_frontend_graph(c)));

  /* An exported and re-imported graph has the same structure. */
  NmGraphModel *imported = export_and_import(a, "tmp_nm_ws_structure.json");
  TEST_ASSERT_EQUAL_UINT64(id_a, nm_graph_structure_id(imported));
  nm_graph_model_free(imported);
  nm_frontend_free(c);
  nm_frontend_free(b);
  nm_frontend_free(a);
}

static void assert_weights_travel(Kind kind) {
  NmFrontendModel *source = make_model(kind, 11);
  NmFrontendModel *other = make_model(kind, 22); /* receiver starts elsewhere */
  NmGraphModel *target = export_and_import(other, "tmp_nm_ws_target.json");

  NmEngineC *sender = nm_engine_c_create(NULL);
  NmEngineC *receiver = nm_engine_c_create(NULL);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare(sender, source));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare_graph(receiver, target));

  FloatMatrix *x = ramp(5, cols_for(kind));
  FloatMatrix *expected = NULL, *before = NULL, *after = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_predict(sender, source, x, &expected));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_predict_graph(receiver, target, x, &before));
  TEST_ASSERT_TRUE(memcmp(expected->values, before->values, 5 * 3 * sizeof(float)) != 0);

  Message m = make_message(nm_frontend_graph(source), 7);
  NmWeightSet *set = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_weightset_open(m.memory, m.size, NM_GIRW_VERIFY_ALL,
                                      release_message, &m, &set));
  TEST_ASSERT_EQUAL_UINT64(7, nm_weightset_version(set));
  TEST_ASSERT_EQUAL_UINT64(nm_graph_structure_id(nm_frontend_graph(source)),
                           nm_weightset_structure_id(set));

  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_bind_weights(receiver, target, set));
  TEST_ASSERT_EQUAL_UINT64(7, nm_graph_bound_version(target));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_predict_graph(receiver, target, x, &after));
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected->values, after->values, 5 * 3);

  /* The receiver's graph reads the message memory directly. */
  const NmGirwEntry *probe = NULL;
  (void)probe;
  nm_weightset_release(set); /* the caller's reference; the graph keeps its own */
  TEST_ASSERT_EQUAL_INT(0, m.released);
  nm_graph_model_free(target);
  TEST_ASSERT_EQUAL_INT(1, m.released); /* handed back exactly once */

  free(m.memory);
  sm_destroy(after);
  sm_destroy(before);
  sm_destroy(expected);
  sm_destroy(x);
  nm_engine_c_free(receiver);
  nm_engine_c_free(sender);
  nm_frontend_free(other);
  nm_frontend_free(source);
}

void test_dense_weights_travel_without_copy(void) { assert_weights_travel(DENSE); }
void test_cnn_weights_travel_without_copy(void) { assert_weights_travel(CNN); }

void test_engine_switches_weights_between_predicts_and_releases_old_sets(void) {
  NmFrontendModel *first = make_model(DENSE, 31);
  NmFrontendModel *second = make_model(DENSE, 32);
  NmGraphModel *target = export_and_import(first, "tmp_nm_ws_switch.json");
  NmEngineC *engine = nm_engine_c_create(NULL);
  NmEngineC *reference = nm_engine_c_create(NULL);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare_graph(engine, target));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare(reference, second));

  Message v1 = make_message(nm_frontend_graph(first), 1);
  Message v2 = make_message(nm_frontend_graph(second), 2);
  NmWeightSet *s1 = NULL, *s2 = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_weightset_open(v1.memory, v1.size, NM_GIRW_VERIFY_ALL,
                                                    release_message, &v1, &s1));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_weightset_open(v2.memory, v2.size, NM_GIRW_VERIFY_ALL,
                                                    release_message, &v2, &s2));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_bind_weights(engine, target, s1));
  nm_weightset_release(s1);
  TEST_ASSERT_EQUAL_INT(0, v1.released); /* still bound */
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_bind_weights(engine, target, s2));
  nm_weightset_release(s2);
  TEST_ASSERT_EQUAL_INT(1, v1.released); /* replaced -> handed back */
  TEST_ASSERT_EQUAL_INT(0, v2.released);
  TEST_ASSERT_EQUAL_UINT64(2, nm_graph_bound_version(target));

  FloatMatrix *x = ramp(4, 8), *got = NULL, *want = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_predict_graph(engine, target, x, &got));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_predict(reference, second, x, &want));
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(want->values, got->values, 4 * 3);

  /* Bound weights are borrowed and read-only: training is refused. */
  FloatMatrix *y = sm_create(4, 3);
  NmTrainConfig config = {.batch_size = 2, .learning_rate = 0.1f, .epochs = 1};
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT,
                    nm_engine_c_fit_graph(engine, target, x, y, &config));

  nm_graph_model_free(target);
  TEST_ASSERT_EQUAL_INT(1, v2.released);
  free(v1.memory);
  free(v2.memory);
  sm_destroy(y);
  sm_destroy(want);
  sm_destroy(got);
  sm_destroy(x);
  nm_engine_c_free(reference);
  nm_engine_c_free(engine);
  nm_frontend_free(second);
  nm_frontend_free(first);
}

void test_weights_for_another_structure_are_rejected_and_graph_is_unchanged(void) {
  NmFrontendModel *cnn = make_model(CNN, 5);
  NmFrontendModel *dense = make_model(DENSE, 5);
  NmGraphModel *target = export_and_import(dense, "tmp_nm_ws_reject.json");
  NmEngineC *engine = nm_engine_c_create(NULL);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare_graph(engine, target));
  FloatMatrix *x = ramp(3, 8), *before = NULL, *after = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_predict_graph(engine, target, x, &before));

  Message m = make_message(nm_frontend_graph(cnn), 1);
  NmWeightSet *set = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_weightset_open(m.memory, m.size, NM_GIRW_VERIFY_ALL,
                                                    release_message, &m, &set));
  TEST_ASSERT_EQUAL(NM_STATUS_FORMAT_ERROR, nm_engine_c_bind_weights(engine, target, set));
  TEST_ASSERT_EQUAL_UINT64(0, nm_graph_bound_version(target));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_predict_graph(engine, target, x, &after));
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(before->values, after->values, 3 * 3);

  nm_weightset_release(set);
  TEST_ASSERT_EQUAL_INT(1, m.released);
  free(m.memory);
  sm_destroy(after);
  sm_destroy(before);
  sm_destroy(x);
  nm_engine_c_free(engine);
  nm_graph_model_free(target);
  nm_frontend_free(dense);
  nm_frontend_free(cnn);
}

void test_damaged_or_misaligned_messages_are_rejected(void) {
  NmFrontendModel *model = make_model(DENSE, 8);
  Message m = make_message(nm_frontend_graph(model), 1);
  NmWeightSet *set = NULL;

  /* Misaligned buffer. */
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT,
                    nm_weightset_open((uint8_t *)m.memory + 8, m.size - 8,
                                      NM_GIRW_VERIFY_ALL, release_message, &m, &set));
  /* Truncated, wrong magic. */
  TEST_ASSERT_EQUAL(NM_STATUS_FORMAT_ERROR,
                    nm_weightset_open(m.memory, 40, NM_GIRW_VERIFY_ALL,
                                      release_message, &m, &set));
  ((uint8_t *)m.memory)[0] ^= 0xFF;
  TEST_ASSERT_EQUAL(NM_STATUS_FORMAT_ERROR,
                    nm_weightset_open(m.memory, m.size, NM_GIRW_VERIFY_ALL,
                                      release_message, &m, &set));
  ((uint8_t *)m.memory)[0] ^= 0xFF;
  /* A flipped weight bit: caught with full verification, tolerated without. */
  ((uint8_t *)m.memory)[m.size - 1] ^= 0x01;
  TEST_ASSERT_EQUAL(NM_STATUS_FORMAT_ERROR,
                    nm_weightset_open(m.memory, m.size, NM_GIRW_VERIFY_ALL,
                                      release_message, &m, &set));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_weightset_open(m.memory, m.size, NM_GIRW_VERIFY_INDEX_ONLY,
                                      release_message, &m, &set));
  TEST_ASSERT_EQUAL_INT(0, m.released); /* never called on failure */
  nm_weightset_release(set);
  TEST_ASSERT_EQUAL_INT(1, m.released);
  free(m.memory);
  nm_frontend_free(model);
}
