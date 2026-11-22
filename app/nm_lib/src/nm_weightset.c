/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm_weightset.h"
#include "nm_graph_ir_internal.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <xxhash.h>

#define HEADER_SIZE 64U
#define FORMAT_VERSION 1U

struct NmWeightSet {
  atomic_size_t references;
  NmGirwPackage *package;
  const uint8_t *message;
  uint64_t epoch;
  uint64_t version;
  uint64_t structure_id;
  NmWeightRelease release;
  void *release_context;
};

typedef struct {
  NmWeightSet *set;
  FloatTensor **wrappers;
  size_t count;
} Binding;

static void put_le(uint8_t *out, size_t width, uint64_t value) {
  for (size_t i = 0; i < width; ++i)
    out[i] = (uint8_t)(value >> (8 * i));
}

static uint64_t get_le(const uint8_t *in, size_t width) {
  uint64_t value = 0;
  for (size_t i = width; i > 0; --i)
    value = (value << 8) | in[i - 1];
  return value;
}

/* ---- Structure id --------------------------------------------------------- */

static void hash_u64(XXH64_state_t *state, uint64_t value) {
  uint8_t bytes[8];
  put_le(bytes, 8, value);
  XXH64_update(state, bytes, 8);
}

static void hash_text(XXH64_state_t *state, const char *text) {
  size_t length = strlen(text);
  hash_u64(state, length);
  XXH64_update(state, text, length);
}

uint64_t nm_graph_structure_id(const NmGraphModel *graph) {
  if (!graph)
    return 0;
  XXH64_state_t *state = XXH64_createState();
  if (!state)
    return 0;
  XXH64_reset(state, 0);
  hash_text(state, "nm.structure.v1");
  hash_u64(state, graph->model_input_id);
  hash_u64(state, graph->model_output_id);
  hash_u64(state, graph->value_count);
  for (size_t i = 0; i < graph->value_count; ++i) {
    const NmIrValue *v = &graph->values[i];
    hash_text(state, v->name);
    hash_u64(state, v->kind);
    hash_u64(state, v->shape.ndim);
    for (size_t d = 0; d < v->shape.ndim; ++d)
      hash_u64(state, v->shape.dims[d]);
  }
  hash_u64(state, graph->node_count);
  for (size_t i = 0; i < graph->node_count; ++i) {
    const NmIrNode *n = &graph->nodes[i];
    hash_u64(state, n->op_type);
    hash_u64(state, n->input_count);
    for (uint32_t k = 0; k < n->input_count; ++k)
      hash_u64(state, n->input_ids[k]);
    hash_u64(state, n->output_id);
    if (n->op_type == NM_IR_OP_CONV2D) {
      hash_u64(state, n->cfg.conv2d.stride_h);
      hash_u64(state, n->cfg.conv2d.stride_w);
      hash_u64(state, n->cfg.conv2d.pad_h);
      hash_u64(state, n->cfg.conv2d.pad_w);
    } else if (n->op_type == NM_IR_OP_MAXPOOL2D) {
      hash_u64(state, n->cfg.maxpool2d.kernel_h);
      hash_u64(state, n->cfg.maxpool2d.kernel_w);
      hash_u64(state, n->cfg.maxpool2d.stride_h);
      hash_u64(state, n->cfg.maxpool2d.stride_w);
      hash_u64(state, n->cfg.maxpool2d.pad_h);
      hash_u64(state, n->cfg.maxpool2d.pad_w);
    }
  }
  uint64_t id = XXH64_digest(state);
  XXH64_freeState(state);
  return id;
}

/* ---- Writing -------------------------------------------------------------- */

static size_t collect_sources(const NmIrGraph *graph, NmGirwSource *sources) {
  size_t count = 0;
  for (size_t i = 0; i < graph->value_count; ++i) {
    const NmIrValue *v = &graph->values[i];
    if (v->kind != NM_IR_VALUE_CONSTANT)
      continue;
    if (!v->data)
      return (size_t)-1; /* e.g. bf16 weights */
    if (sources) {
      sources[count] = (NmGirwSource){.name = v->name,
                                      .dtype = NM_GIRW_FLOAT32,
                                      .element_count = v->data_count,
                                      .data = v->data};
    }
    ++count;
  }
  return count;
}

size_t nm_weightset_message_size(const NmGraphModel *graph) {
  if (!graph)
    return 0;
  size_t count = collect_sources(graph, NULL);
  if (count == 0 || count == (size_t)-1)
    return 0;
  NmGirwSource *sources = calloc(count, sizeof(*sources));
  if (!sources)
    return 0;
  collect_sources(graph, sources);
  size_t package = nm_girw_image_size(sources, count);
  free(sources);
  return package ? HEADER_SIZE + package : 0;
}

NmStatus nm_weightset_write(const NmGraphModel *graph, uint64_t epoch,
                            uint64_t version, void *out, size_t out_size) {
  if (!graph || !out || ((uintptr_t)out % NM_WEIGHTSET_ALIGNMENT) != 0)
    return NM_STATUS_INVALID_ARGUMENT;
  size_t count = collect_sources(graph, NULL);
  if (count == 0)
    return NM_STATUS_INVALID_ARGUMENT;
  if (count == (size_t)-1)
    return NM_STATUS_UNSUPPORTED_TYPE;
  NmGirwSource *sources = calloc(count, sizeof(*sources));
  if (!sources)
    return NM_STATUS_ALLOC_FAILED;
  collect_sources(graph, sources);
  size_t package = nm_girw_image_size(sources, count);
  NmStatus status = NM_STATUS_INVALID_ARGUMENT;
  if (package && out_size >= HEADER_SIZE + package) {
    uint8_t *bytes = out;
    status = nm_girw_write_image(sources, count, bytes + HEADER_SIZE, package);
    if (status == NM_STATUS_OK) {
      memset(bytes, 0, HEADER_SIZE);
      memcpy(bytes, "NMWSET01", 8);
      put_le(bytes + 8, 4, FORMAT_VERSION);
      put_le(bytes + 12, 4, HEADER_SIZE);
      put_le(bytes + 16, 8, version);
      put_le(bytes + 24, 8, nm_graph_structure_id(graph));
      put_le(bytes + 32, 8, package);
      put_le(bytes + 40, 8, epoch);
    }
  }
  free(sources);
  return status;
}

/* ---- Reading -------------------------------------------------------------- */

NmStatus nm_weightset_open(const void *message, size_t size, NmGirwVerify verify,
                           NmWeightRelease release, void *release_context,
                           NmWeightSet **out_set) {
  if (!message || !out_set)
    return NM_STATUS_INVALID_ARGUMENT;
  *out_set = NULL;
  if (((uintptr_t)message % NM_WEIGHTSET_ALIGNMENT) != 0)
    return NM_STATUS_INVALID_ARGUMENT;
  const uint8_t *bytes = message;
  if (size < HEADER_SIZE || memcmp(bytes, "NMWSET01", 8) != 0 ||
      get_le(bytes + 8, 4) != FORMAT_VERSION || get_le(bytes + 12, 4) != HEADER_SIZE)
    return NM_STATUS_FORMAT_ERROR;
  uint64_t package_size = get_le(bytes + 32, 8);
  if (package_size > size - HEADER_SIZE)
    return NM_STATUS_FORMAT_ERROR;

  NmWeightSet *set = calloc(1, sizeof(*set));
  if (!set)
    return NM_STATUS_ALLOC_FAILED;
  NmStatus status = nm_girw_open_memory(bytes + HEADER_SIZE, (size_t)package_size,
                                        verify, &set->package);
  if (status != NM_STATUS_OK) {
    free(set);
    return status;
  }
  atomic_init(&set->references, 1);
  set->message = bytes;
  set->version = get_le(bytes + 16, 8);
  set->epoch = get_le(bytes + 40, 8);
  set->structure_id = get_le(bytes + 24, 8);
  set->release = release;
  set->release_context = release_context;
  *out_set = set;
  return NM_STATUS_OK;
}

uint64_t nm_weightset_epoch(const NmWeightSet *set) {
  return set ? set->epoch : 0;
}

uint64_t nm_weightset_version(const NmWeightSet *set) {
  return set ? set->version : 0;
}

uint64_t nm_weightset_structure_id(const NmWeightSet *set) {
  return set ? set->structure_id : 0;
}

void nm_weightset_retain(NmWeightSet *set) {
  if (set)
    atomic_fetch_add(&set->references, 1);
}

void nm_weightset_release(NmWeightSet *set) {
  if (!set || atomic_fetch_sub(&set->references, 1) != 1)
    return;
  nm_girw_close(set->package);
  if (set->release)
    set->release(set->release_context);
  free(set);
}

/* ---- Binding -------------------------------------------------------------- */

static void binding_free(void *pointer) {
  Binding *binding = pointer;
  for (size_t i = 0; i < binding->count; ++i)
    st_destroy(binding->wrappers[i]);
  free(binding->wrappers);
  nm_weightset_release(binding->set);
  free(binding);
}

static const NmGirwEntry *find_entry(const NmGirwPackage *package, const char *name) {
  for (size_t i = 0; i < nm_girw_entry_count(package); ++i) {
    const NmGirwEntry *entry = nm_girw_entry(package, i);
    if (!strcmp(entry->name, name))
      return entry;
  }
  return NULL;
}

static bool wants_tensor(const NmIrValue *v) {
  return v->shape.ndim == 1 || v->shape.ndim == 4;
}

NmStatus nm_graph_bind_weights(NmGraphModel *graph, NmWeightSet *set) {
  if (!graph || !set)
    return NM_STATUS_INVALID_ARGUMENT;
  if (set->structure_id != nm_graph_structure_id(graph))
    return NM_STATUS_FORMAT_ERROR;

  /* Validate everything before touching the graph. */
  size_t constants = 0, tensors = 0;
  for (size_t i = 0; i < graph->value_count; ++i) {
    const NmIrValue *v = &graph->values[i];
    if (v->kind != NM_IR_VALUE_CONSTANT)
      continue;
    const NmGirwEntry *e = find_entry(set->package, v->name);
    size_t count = 1;
    for (size_t d = 0; d < v->shape.ndim; ++d)
      count *= v->shape.dims[d];
    if (!e || e->dtype != NM_GIRW_FLOAT32 || e->element_count != count ||
        ((uintptr_t)e->data % 16) != 0)
      return NM_STATUS_FORMAT_ERROR;
    ++constants;
    tensors += wants_tensor(v);
  }
  if (constants == 0)
    return NM_STATUS_INVALID_ARGUMENT;

  Binding *binding = calloc(1, sizeof(*binding));
  FloatTensor **wrappers = calloc(tensors ? tensors : 1, sizeof(*wrappers));
  if (!binding || !wrappers) {
    free(binding);
    free(wrappers);
    return NM_STATUS_ALLOC_FAILED;
  }
  binding->wrappers = wrappers;
  size_t made = 0;
  for (size_t i = 0; i < graph->value_count; ++i) {
    const NmIrValue *v = &graph->values[i];
    if (v->kind != NM_IR_VALUE_CONSTANT || !wants_tensor(v))
      continue;
    const NmGirwEntry *e = find_entry(set->package, v->name);
    wrappers[made] = st_create_with_data(v->shape.ndim, v->shape.dims,
                                         (float *)(uintptr_t)e->data,
                                         (size_t)e->element_count, false);
    if (!wrappers[made]) {
      binding->count = made;
      binding_free(binding); /* set not retained yet: undo only the wrappers */
      return NM_STATUS_ALLOC_FAILED;
    }
    ++made;
  }
  binding->count = made;
  binding->set = set;
  nm_weightset_retain(set);

  /* Commit: no failure is possible from here on. */
  size_t next_wrapper = 0;
  for (size_t i = 0; i < graph->value_count; ++i) {
    NmIrValue *v = &graph->values[i];
    if (v->kind != NM_IR_VALUE_CONSTANT)
      continue;
    const NmGirwEntry *e = find_entry(set->package, v->name);
    if (v->owns_data) {
      if (v->tensor)
        st_destroy(v->tensor);
      else
        free(v->data);
    }
    v->data = (float *)(uintptr_t)e->data;
    v->data_count = (size_t)e->element_count;
    v->tensor = wants_tensor(v) ? wrappers[next_wrapper++] : NULL;
    v->owns_data = false;
  }
  if (graph->binding && graph->binding_free)
    graph->binding_free(graph->binding);
  graph->binding = binding;
  graph->binding_free = binding_free;
  return NM_STATUS_OK;
}

uint64_t nm_graph_bound_version(const NmGraphModel *graph) {
  if (!graph || !graph->binding)
    return 0;
  return ((const Binding *)graph->binding)->set->version;
}

uint64_t nm_graph_bound_epoch(const NmGraphModel *graph) {
  if (!graph || !graph->binding)
    return 0;
  return ((const Binding *)graph->binding)->set->epoch;
}
