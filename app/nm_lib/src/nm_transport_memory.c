/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* In-process transport: a bounded queue of aligned heap messages. */
#include "nm_connector.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  void *buffer;
  size_t size;
} Slot;

typedef struct {
  pthread_mutex_t lock;
  Slot *slots;
  size_t capacity;
  size_t head;
  size_t count;
  int open_ends;
} Queue;

static void free_buffer(void *buffer) { free(buffer); }

static NmStatus memory_loan(void *context, size_t size, void **buffer) {
  (void)context;
  void *memory = NULL;
  if (size == 0 || posix_memalign(&memory, NM_WEIGHTSET_ALIGNMENT,
                                  (size + 63) / 64 * 64) != 0)
    return NM_STATUS_ALLOC_FAILED;
  *buffer = memory;
  return NM_STATUS_OK;
}

static void memory_discard(void *context, void *buffer) {
  (void)context;
  free(buffer);
}

static NmStatus memory_send(void *context, void *buffer, size_t size) {
  Queue *q = context;
  pthread_mutex_lock(&q->lock);
  if (q->count == q->capacity) { /* drop the oldest */
    free(q->slots[q->head].buffer);
    q->head = (q->head + 1) % q->capacity;
    q->count -= 1;
  }
  q->slots[(q->head + q->count) % q->capacity] = (Slot){buffer, size};
  q->count += 1;
  pthread_mutex_unlock(&q->lock);
  return NM_STATUS_OK;
}

static NmStatus memory_receive(void *context, const void **message, size_t *size,
                               NmWeightRelease *release, void **release_context) {
  Queue *q = context;
  *message = NULL;
  pthread_mutex_lock(&q->lock);
  if (q->count > 0) {
    Slot slot = q->slots[q->head];
    q->head = (q->head + 1) % q->capacity;
    q->count -= 1;
    *message = slot.buffer;
    *size = slot.size;
    *release = free_buffer;
    *release_context = slot.buffer;
  }
  pthread_mutex_unlock(&q->lock);
  return NM_STATUS_OK;
}

static void memory_close(void *context) {
  Queue *q = context;
  pthread_mutex_lock(&q->lock);
  int remaining = --q->open_ends;
  pthread_mutex_unlock(&q->lock);
  if (remaining > 0)
    return;
  for (size_t i = 0; i < q->count; ++i)
    free(q->slots[(q->head + i) % q->capacity].buffer);
  pthread_mutex_destroy(&q->lock);
  free(q->slots);
  free(q);
}

NmStatus nm_transport_memory_pair(size_t capacity, NmTransport *sender,
                                  NmTransport *receiver) {
  if (!capacity || !sender || !receiver)
    return NM_STATUS_INVALID_ARGUMENT;
  Queue *q = calloc(1, sizeof(*q));
  if (!q)
    return NM_STATUS_ALLOC_FAILED;
  q->slots = calloc(capacity, sizeof(*q->slots));
  if (!q->slots) {
    free(q);
    return NM_STATUS_ALLOC_FAILED;
  }
  q->capacity = capacity;
  q->open_ends = 2;
  pthread_mutex_init(&q->lock, NULL);
  *sender = (NmTransport){.context = q,
                          .loan = memory_loan,
                          .send = memory_send,
                          .discard = memory_discard,
                          .close = memory_close};
  *receiver = (NmTransport){.context = q, .receive = memory_receive, .close = memory_close};
  return NM_STATUS_OK;
}
