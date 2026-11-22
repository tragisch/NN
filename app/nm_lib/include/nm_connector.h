/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Connectors move the weights of a graph between processes (or components) and
 * keep a running engine up to date.
 *
 * Layers, from the bottom:
 *   transport  moves opaque 64-byte aligned messages (NmTransport)
 *   message    a weight set (nm_weightset.h): .girw package + header
 *   binding    nm_engine_c_bind_weights, zero-copy
 *   control    publisher/subscriber below: versions, structure checks, loss
 *
 * Transports in this library: an in-process queue and a directory of files.
 * Others (e.g. shared memory) plug in through the same table. */
#ifndef NM_CONNECTOR_H
#define NM_CONNECTOR_H

#include "nm_engine_c.h"
#include "nm_weightset.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NmTransport {
  void *context;
  /* Sender: borrow a writable, 64-byte aligned buffer of `size` bytes, then
   * either send it (the transport takes it over) or discard it. */
  NmStatus (*loan)(void *context, size_t size, void **buffer);
  NmStatus (*send)(void *context, void *buffer, size_t size);
  void (*discard)(void *context, void *buffer);
  /* Receiver: hands out the next message or sets *message to NULL. The message
   * is valid until `*release(*release_context)` is called. */
  NmStatus (*receive)(void *context, const void **message, size_t *size,
                      NmWeightRelease *release, void **release_context);
  /* Keeps connections alive; call it regularly (a late receiver is only
   * connected once the sender maintains its connections). May be NULL. */
  NmStatus (*maintain)(void *context);
  void (*close)(void *context);
} NmTransport;

/* ---- Transports ----------------------------------------------------------- */

/* A queue inside one process. `capacity` messages are kept; when it is full the
 * oldest is dropped. Both ends share the queue; closing both frees it. */
NmStatus nm_transport_memory_pair(size_t capacity, NmTransport *sender,
                                  NmTransport *receiver);

/* Messages as files in `directory` (created if missing): the sender writes
 * `weights.<n>.nmws` atomically and keeps the newest three; the receiver reads
 * files it has not seen yet, oldest first. For tests and simple deployments. */
NmStatus nm_transport_file_sender(const char *directory, NmTransport *sender);
NmStatus nm_transport_file_receiver(const char *directory, NmTransport *receiver);

/* ---- Publisher ------------------------------------------------------------ */

typedef struct NmConnectorPublisher NmConnectorPublisher;

/* Takes over `transport`. The graph must stay valid while publishing. */
NmConnectorPublisher *nm_connector_publisher_create(NmTransport transport,
                                                    const NmGraphModel *graph);
void nm_connector_publisher_free(NmConnectorPublisher *publisher);

/* Publishes the graph's current weights as the next version. */
NmStatus nm_connector_publisher_publish(NmConnectorPublisher *publisher);
NmStatus nm_connector_publisher_maintain(NmConnectorPublisher *publisher);
uint64_t nm_connector_publisher_version(const NmConnectorPublisher *publisher);

/* ---- Subscriber ----------------------------------------------------------- */

typedef enum {
  NM_CONNECTOR_WAITING = 0,   /* no weights bound yet */
  NM_CONNECTOR_BOUND = 1,     /* running on received weights */
  NM_CONNECTOR_STALE = 2,     /* publisher silent for too long, last weights stay */
  NM_CONNECTOR_REJECTING = 3, /* the last message did not fit; previous weights stay */
} NmConnectorState;

typedef struct {
  NmGirwVerify verify;       /* default NM_GIRW_VERIFY_ALL */
  uint64_t stale_after_ns;   /* 0: never stale */
  uint64_t (*clock_ns)(void *context); /* NULL: CLOCK_MONOTONIC */
  void *clock_context;
} NmConnectorConfig;

typedef struct NmConnectorSubscriber NmConnectorSubscriber;

/* Takes over `transport`. `engine` must be prepared for `graph`. */
NmConnectorSubscriber *nm_connector_subscriber_create(NmTransport transport,
                                                      NmEngineC *engine,
                                                      NmGraphModel *graph,
                                                      const NmConnectorConfig *config);
void nm_connector_subscriber_free(NmConnectorSubscriber *subscriber);

/* Call between predicts. Maintains the transport, takes the newest message and
 * binds it if it fits and is newer than what is bound. Returns the transport's
 * status; a message that does not fit is not an error but counts as rejected. */
NmStatus nm_connector_subscriber_poll(NmConnectorSubscriber *subscriber);

NmConnectorState nm_connector_subscriber_state(const NmConnectorSubscriber *subscriber);
uint64_t nm_connector_subscriber_bound_version(const NmConnectorSubscriber *subscriber);
size_t nm_connector_subscriber_rejected(const NmConnectorSubscriber *subscriber);
size_t nm_connector_subscriber_bound_count(const NmConnectorSubscriber *subscriber);

#ifdef __cplusplus
}
#endif

#endif /* NM_CONNECTOR_H */
