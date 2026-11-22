/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm_connector.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ---- Publisher ------------------------------------------------------------ */

struct NmConnectorPublisher {
  NmTransport transport;
  const NmGraphModel *graph;
  uint64_t epoch;
  uint64_t version;
};

static uint64_t monotonic_ns(void *context) {
  (void)context;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint64_t make_epoch(void) {
  static uint64_t counter;
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  uint64_t epoch = ((uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec) ^
                   ((uint64_t)getpid() << 40) ^ (++counter * 0x9E3779B97F4A7C15ULL);
  return epoch ? epoch : 1;
}

NmConnectorPublisher *nm_connector_publisher_create(NmTransport transport,
                                                    const NmGraphModel *graph) {
  if (!graph || !transport.loan || !transport.send)
    return NULL;
  NmConnectorPublisher *publisher = calloc(1, sizeof(*publisher));
  if (!publisher)
    return NULL;
  publisher->transport = transport;
  publisher->graph = graph;
  publisher->epoch = make_epoch();
  return publisher;
}

void nm_connector_publisher_free(NmConnectorPublisher *publisher) {
  if (!publisher)
    return;
  if (publisher->transport.close)
    publisher->transport.close(publisher->transport.context);
  free(publisher);
}

NmStatus nm_connector_publisher_publish(NmConnectorPublisher *publisher) {
  if (!publisher)
    return NM_STATUS_INVALID_ARGUMENT;
  size_t size = nm_weightset_message_size(publisher->graph);
  if (size == 0)
    return NM_STATUS_UNSUPPORTED_TYPE;
  void *buffer = NULL;
  NmStatus status =
      publisher->transport.loan(publisher->transport.context, size, &buffer);
  if (status != NM_STATUS_OK)
    return status;
  status = nm_weightset_write(publisher->graph, publisher->epoch,
                              publisher->version + 1, buffer, size);
  if (status != NM_STATUS_OK) {
    if (publisher->transport.discard)
      publisher->transport.discard(publisher->transport.context, buffer);
    return status;
  }
  status = publisher->transport.send(publisher->transport.context, buffer, size);
  if (status == NM_STATUS_OK)
    publisher->version += 1;
  return status;
}

NmStatus nm_connector_publisher_maintain(NmConnectorPublisher *publisher) {
  if (!publisher)
    return NM_STATUS_INVALID_ARGUMENT;
  return publisher->transport.maintain
             ? publisher->transport.maintain(publisher->transport.context)
             : NM_STATUS_OK;
}

uint64_t nm_connector_publisher_version(const NmConnectorPublisher *publisher) {
  return publisher ? publisher->version : 0;
}

/* ---- Subscriber ----------------------------------------------------------- */

struct NmConnectorSubscriber {
  NmTransport transport;
  NmEngineC *engine;
  NmGraphModel *graph;
  NmConnectorConfig config;
  NmConnectorState state;
  uint64_t last_message_ns;
  size_t rejected;
  size_t bound_count;
};

NmConnectorSubscriber *nm_connector_subscriber_create(NmTransport transport,
                                                      NmEngineC *engine,
                                                      NmGraphModel *graph,
                                                      const NmConnectorConfig *config) {
  if (!engine || !graph || !transport.receive)
    return NULL;
  NmConnectorSubscriber *s = calloc(1, sizeof(*s));
  if (!s)
    return NULL;
  s->transport = transport;
  s->engine = engine;
  s->graph = graph;
  if (config)
    s->config = *config;
  if (!s->config.clock_ns)
    s->config.clock_ns = monotonic_ns;
  s->state = NM_CONNECTOR_WAITING;
  s->last_message_ns = s->config.clock_ns(s->config.clock_context);
  return s;
}

void nm_connector_subscriber_free(NmConnectorSubscriber *s) {
  if (!s)
    return;
  if (s->transport.close)
    s->transport.close(s->transport.context);
  free(s);
}

/* Is `set` newer than what the graph is bound to? */
static bool newer_than_bound(const NmConnectorSubscriber *s, const NmWeightSet *set) {
  if (nm_graph_bound_version(s->graph) == 0)
    return true;
  return nm_weightset_epoch(set) != nm_graph_bound_epoch(s->graph) ||
         nm_weightset_version(set) > nm_graph_bound_version(s->graph);
}

NmStatus nm_connector_subscriber_poll(NmConnectorSubscriber *s) {
  if (!s)
    return NM_STATUS_INVALID_ARGUMENT;
  NmStatus transport_status = NM_STATUS_OK;
  if (s->transport.maintain)
    transport_status = s->transport.maintain(s->transport.context);

  /* Drain the transport and keep the newest message that opens cleanly. */
  NmWeightSet *candidate = NULL;
  bool received = false, damaged = false;
  for (;;) {
    const void *message = NULL;
    size_t size = 0;
    NmWeightRelease release = NULL;
    void *release_context = NULL;
    NmStatus status = s->transport.receive(s->transport.context, &message, &size,
                                           &release, &release_context);
    if (status != NM_STATUS_OK) {
      transport_status = status;
      break;
    }
    if (!message)
      break;
    received = true;
    NmWeightSet *set = NULL;
    status = nm_weightset_open(message, size, s->config.verify, release,
                               release_context, &set);
    if (status != NM_STATUS_OK) {
      damaged = true;
      if (release)
        release(release_context);
      continue;
    }
    if (!candidate || nm_weightset_epoch(set) != nm_weightset_epoch(candidate) ||
        nm_weightset_version(set) > nm_weightset_version(candidate)) {
      nm_weightset_release(candidate);
      candidate = set;
    } else {
      nm_weightset_release(set);
    }
  }

  const uint64_t now = s->config.clock_ns(s->config.clock_context);
  if (received)
    s->last_message_ns = now;

  if (candidate && newer_than_bound(s, candidate)) {
    NmStatus status = nm_engine_c_bind_weights(s->engine, s->graph, candidate);
    if (status == NM_STATUS_OK) {
      s->state = NM_CONNECTOR_BOUND;
      s->bound_count += 1;
      damaged = false;
    } else {
      s->rejected += 1;
      s->state = NM_CONNECTOR_REJECTING;
    }
  } else if (damaged) {
    s->rejected += 1;
    s->state = NM_CONNECTOR_REJECTING;
  }
  nm_weightset_release(candidate);

  if (s->config.stale_after_ns && s->state == NM_CONNECTOR_BOUND &&
      now - s->last_message_ns > s->config.stale_after_ns)
    s->state = NM_CONNECTOR_STALE;
  return transport_status;
}

NmConnectorState nm_connector_subscriber_state(const NmConnectorSubscriber *s) {
  return s ? s->state : NM_CONNECTOR_WAITING;
}

uint64_t nm_connector_subscriber_bound_version(const NmConnectorSubscriber *s) {
  return s ? nm_graph_bound_version(s->graph) : 0;
}

size_t nm_connector_subscriber_rejected(const NmConnectorSubscriber *s) {
  return s ? s->rejected : 0;
}

size_t nm_connector_subscriber_bound_count(const NmConnectorSubscriber *s) {
  return s ? s->bound_count : 0;
}
