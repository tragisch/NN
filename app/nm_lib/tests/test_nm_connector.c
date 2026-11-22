/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm_connector.h"
#include "nm_export_graphir.h"
#include "nm_import_graphir.h"
#include "nm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define UNITY_INCLUDE_FLOAT
#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static NmFrontendModel *make_model(uint64_t seed, size_t hidden) {
  nm_set_seed(seed);
  NmFrontendModel *m = nm_frontend_create((NmShape){.ndim = 1, .dims = {8}});
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_frontend_add_dense(
                        m, &(NmDenseConfig){.units = hidden, .activation = NM_ACT_RELU}));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_frontend_add_dense(
                        m, &(NmDenseConfig){.units = 3, .activation = NM_ACT_SOFTMAX}));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_frontend_finalize(m));
  return m;
}

static FloatMatrix *ramp(size_t rows) {
  FloatMatrix *x = sm_create(rows, 8);
  for (size_t i = 0; i < rows * 8; ++i)
    x->values[i] = (float)((int)(i * 7U % 13U) - 6) / 6.0f;
  return x;
}

/* Receiver side: a graph with the same structure, imported from the sender. */
typedef struct {
  NmGraphModel *graph;
  NmEngineC *engine;
} Receiver;

static Receiver make_receiver(NmFrontendModel *like, const char *path) {
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_export_graphir_json(
                        like, &(NmGraphIrExportOptions){.batch_size = 1}, path));
  Receiver r = {0};
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_import_graphir_json(path, &r.graph));
  remove(path);
  r.engine = nm_engine_c_create(NULL);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare_graph(r.engine, r.graph));
  return r;
}

/* A receiver whose graph starts from another model's weights. */
static Receiver make_receiver_from_seed(uint64_t seed, size_t hidden, const char *path) {
  NmFrontendModel *like = make_model(seed, hidden);
  Receiver r = make_receiver(like, path);
  nm_frontend_free(like);
  return r;
}

static void free_receiver(Receiver *r) {
  nm_engine_c_free(r->engine);
  nm_graph_model_free(r->graph);
}

static void assert_same_prediction(NmEngineC *a, NmFrontendModel *model,
                                   NmEngineC *b, NmGraphModel *graph) {
  FloatMatrix *x = ramp(4), *want = NULL, *got = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_predict(a, model, x, &want));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_predict_graph(b, graph, x, &got));
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(want->values, got->values, 4 * 3);
  sm_destroy(got);
  sm_destroy(want);
  sm_destroy(x);
}

/* A manually advanced clock. */
static uint64_t g_now;
static uint64_t fake_clock(void *context) {
  (void)context;
  return g_now;
}

void test_subscriber_follows_the_publisher_through_memory(void) {
  NmFrontendModel *source = make_model(1, 6);
  Receiver r = make_receiver_from_seed(2, 6, "tmp_nm_conn_mem.json");
  NmEngineC *sender_engine = nm_engine_c_create(NULL);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare(sender_engine, source));

  NmTransport tx, rx;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_transport_memory_pair(4, &tx, &rx));
  NmConnectorPublisher *pub = nm_connector_publisher_create(tx, nm_frontend_graph(source));
  NmConnectorSubscriber *sub = nm_connector_subscriber_create(rx, r.engine, r.graph, NULL);

  TEST_ASSERT_EQUAL(NM_CONNECTOR_WAITING, nm_connector_subscriber_state(sub));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_subscriber_poll(sub));
  TEST_ASSERT_EQUAL(NM_CONNECTOR_WAITING, nm_connector_subscriber_state(sub));

  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_publisher_publish(pub));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_subscriber_poll(sub));
  TEST_ASSERT_EQUAL(NM_CONNECTOR_BOUND, nm_connector_subscriber_state(sub));
  TEST_ASSERT_EQUAL_UINT64(1, nm_connector_subscriber_bound_version(sub));
  assert_same_prediction(sender_engine, source, r.engine, r.graph);

  /* The sender trains; two publishes before the next poll: the newest wins. */
  FloatMatrix *x = ramp(8), *y = sm_create(8, 3);
  for (size_t i = 0; i < 8; ++i)
    y->values[i * 3 + i % 3] = 1.0f;
  NmTrainConfig config = {.batch_size = 4, .learning_rate = 0.1f, .epochs = 3};
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_fit(sender_engine, source, x, y, &config));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_publisher_publish(pub));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_fit(sender_engine, source, x, y, &config));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_publisher_publish(pub));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_subscriber_poll(sub));
  TEST_ASSERT_EQUAL_UINT64(3, nm_connector_subscriber_bound_version(sub));
  TEST_ASSERT_EQUAL_size_t(2, nm_connector_subscriber_bound_count(sub)); /* v1, v3 */
  assert_same_prediction(sender_engine, source, r.engine, r.graph);

  nm_connector_subscriber_free(sub);
  nm_connector_publisher_free(pub);
  sm_destroy(y);
  sm_destroy(x);
  free_receiver(&r);
  nm_engine_c_free(sender_engine);
  nm_frontend_free(source);
}

void test_a_restarted_publisher_is_followed_even_though_versions_start_over(void) {
  NmFrontendModel *first = make_model(3, 6);
  NmFrontendModel *second = make_model(4, 6);
  Receiver r = make_receiver(first, "tmp_nm_conn_restart.json");
  NmEngineC *engine_second = nm_engine_c_create(NULL);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare(engine_second, second));

  NmTransport tx1, rx1;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_transport_memory_pair(4, &tx1, &rx1));
  NmConnectorPublisher *pub1 = nm_connector_publisher_create(tx1, nm_frontend_graph(first));
  NmConnectorSubscriber *sub = nm_connector_subscriber_create(rx1, r.engine, r.graph, NULL);
  for (int i = 0; i < 3; ++i)
    TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_publisher_publish(pub1));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_subscriber_poll(sub));
  TEST_ASSERT_EQUAL_UINT64(3, nm_connector_subscriber_bound_version(sub));
  nm_connector_subscriber_free(sub);
  nm_connector_publisher_free(pub1);

  /* A new publisher run starts at version 1 again, with different weights. */
  NmTransport tx2, rx2;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_transport_memory_pair(4, &tx2, &rx2));
  NmConnectorPublisher *pub2 = nm_connector_publisher_create(tx2, nm_frontend_graph(second));
  sub = nm_connector_subscriber_create(rx2, r.engine, r.graph, NULL);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_publisher_publish(pub2));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_subscriber_poll(sub));
  TEST_ASSERT_EQUAL_UINT64(1, nm_connector_subscriber_bound_version(sub));
  assert_same_prediction(engine_second, second, r.engine, r.graph);

  nm_connector_subscriber_free(sub);
  nm_connector_publisher_free(pub2);
  free_receiver(&r);
  nm_engine_c_free(engine_second);
  nm_frontend_free(second);
  nm_frontend_free(first);
}

void test_weights_for_another_structure_are_rejected_and_old_weights_keep_running(void) {
  NmFrontendModel *source = make_model(5, 6);
  NmFrontendModel *wrong = make_model(6, 9); /* other hidden size */
  Receiver r = make_receiver_from_seed(7, 6, "tmp_nm_conn_reject.json");
  NmEngineC *want_engine = nm_engine_c_create(NULL);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare(want_engine, source));

  NmTransport tx, rx;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_transport_memory_pair(4, &tx, &rx));
  NmConnectorPublisher *good = nm_connector_publisher_create(tx, nm_frontend_graph(source));
  NmConnectorSubscriber *sub = nm_connector_subscriber_create(rx, r.engine, r.graph, NULL);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_publisher_publish(good));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_subscriber_poll(sub));
  TEST_ASSERT_EQUAL(NM_CONNECTOR_BOUND, nm_connector_subscriber_state(sub));

  /* Same queue, but the next publisher belongs to a different model. */
  NmTransport tx2, rx2;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_transport_memory_pair(4, &tx2, &rx2));
  nm_connector_subscriber_free(sub);
  sub = nm_connector_subscriber_create(rx2, r.engine, r.graph, NULL);
  NmConnectorPublisher *bad = nm_connector_publisher_create(tx2, nm_frontend_graph(wrong));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_publisher_publish(bad));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_subscriber_poll(sub));
  TEST_ASSERT_EQUAL(NM_CONNECTOR_REJECTING, nm_connector_subscriber_state(sub));
  TEST_ASSERT_EQUAL_size_t(1, nm_connector_subscriber_rejected(sub));
  assert_same_prediction(want_engine, source, r.engine, r.graph); /* unchanged */

  nm_connector_subscriber_free(sub);
  nm_connector_publisher_free(bad);
  nm_connector_publisher_free(good);
  free_receiver(&r);
  nm_engine_c_free(want_engine);
  nm_frontend_free(wrong);
  nm_frontend_free(source);
}

void test_silent_publisher_makes_the_subscriber_stale_but_it_keeps_computing(void) {
  NmFrontendModel *source = make_model(8, 6);
  Receiver r = make_receiver_from_seed(9, 6, "tmp_nm_conn_stale.json");
  NmEngineC *want_engine = nm_engine_c_create(NULL);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare(want_engine, source));

  NmTransport tx, rx;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_transport_memory_pair(4, &tx, &rx));
  NmConnectorPublisher *pub = nm_connector_publisher_create(tx, nm_frontend_graph(source));
  NmConnectorConfig config = {.stale_after_ns = 1000, .clock_ns = fake_clock};
  g_now = 100;
  NmConnectorSubscriber *sub = nm_connector_subscriber_create(rx, r.engine, r.graph, &config);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_publisher_publish(pub));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_subscriber_poll(sub));
  TEST_ASSERT_EQUAL(NM_CONNECTOR_BOUND, nm_connector_subscriber_state(sub));

  g_now = 900;
  nm_connector_subscriber_poll(sub);
  TEST_ASSERT_EQUAL(NM_CONNECTOR_BOUND, nm_connector_subscriber_state(sub));
  g_now = 5000;
  nm_connector_subscriber_poll(sub);
  TEST_ASSERT_EQUAL(NM_CONNECTOR_STALE, nm_connector_subscriber_state(sub));
  assert_same_prediction(want_engine, source, r.engine, r.graph); /* still computing */

  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_publisher_publish(pub));
  nm_connector_subscriber_poll(sub);
  TEST_ASSERT_EQUAL(NM_CONNECTOR_BOUND, nm_connector_subscriber_state(sub));

  nm_connector_subscriber_free(sub);
  nm_connector_publisher_free(pub);
  free_receiver(&r);
  nm_engine_c_free(want_engine);
  nm_frontend_free(source);
}

void test_file_transport_moves_weights_between_publisher_and_a_late_subscriber(void) {
  char directory[] = "tmp_nm_conn_dir_XXXXXX";
  TEST_ASSERT_NOT_NULL(mkdtemp(directory));
  NmFrontendModel *source = make_model(10, 6);
  Receiver r = make_receiver_from_seed(11, 6, "tmp_nm_conn_file.json");
  NmEngineC *want_engine = nm_engine_c_create(NULL);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_engine_c_prepare(want_engine, source));

  NmTransport tx;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_transport_file_sender(directory, &tx));
  NmConnectorPublisher *pub = nm_connector_publisher_create(tx, nm_frontend_graph(source));
  for (int i = 0; i < 5; ++i) /* only the newest three files are kept */
    TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_publisher_publish(pub));

  NmTransport rx;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_transport_file_receiver(directory, &rx));
  NmConnectorSubscriber *sub = nm_connector_subscriber_create(rx, r.engine, r.graph, NULL);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_subscriber_poll(sub));
  TEST_ASSERT_EQUAL(NM_CONNECTOR_BOUND, nm_connector_subscriber_state(sub));
  TEST_ASSERT_EQUAL_UINT64(5, nm_connector_subscriber_bound_version(sub));
  assert_same_prediction(want_engine, source, r.engine, r.graph);

  /* Nothing new: the file is not handed out twice. */
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_connector_subscriber_poll(sub));
  TEST_ASSERT_EQUAL_size_t(1, nm_connector_subscriber_bound_count(sub));

  nm_connector_subscriber_free(sub);
  nm_connector_publisher_free(pub);
  free_receiver(&r);
  nm_engine_c_free(want_engine);
  nm_frontend_free(source);
  char command[160];
  snprintf(command, sizeof(command), "rm -rf %s", directory);
  TEST_ASSERT_EQUAL_INT(0, system(command));
}
