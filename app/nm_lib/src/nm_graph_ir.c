/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/*
 * Internal graph-IR helpers for nm models.
 */

#include "nm_graph_ir_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void release_value(NmIrValue *value) {
  free(value->raw);
  value->raw = NULL;
  if (value->owns_data) {
    if (value->tensor)
      st_destroy(value->tensor);
    else
      free(value->data);
  }
  value->data = NULL;
  value->tensor = NULL;
  value->owns_data = false;
}

static void release_values(NmIrGraph *graph) {
  for (size_t i = 0; i < graph->value_count; ++i)
    release_value(&graph->values[i]);
  if (graph->binding && graph->binding_free)
    graph->binding_free(graph->binding);
  graph->binding = NULL;
  graph->binding_free = NULL;
}

bool nm_ir_graph_init(NmIrGraph *graph) {
  if (!graph)
    return false;
  memset(graph, 0, sizeof(*graph));
  graph->model_input_id = NM_IR_INVALID_ID;
  graph->model_output_id = NM_IR_INVALID_ID;
  return true;
}

void nm_ir_graph_reset(NmIrGraph *graph) {
  if (!graph)
    return;
  release_values(graph);
  graph->value_count = 0;
  graph->node_count = 0;
  graph->input_count = 0;
  graph->model_input_id = NM_IR_INVALID_ID;
  graph->model_output_id = NM_IR_INVALID_ID;
}

void nm_ir_graph_destroy(NmIrGraph *graph) {
  if (!graph)
    return;
  release_values(graph);
  free(graph->values);
  free(graph->nodes);
  graph->values = NULL;
  graph->nodes = NULL;
  graph->value_count = 0;
  graph->value_capacity = 0;
  graph->node_count = 0;
  graph->node_capacity = 0;
  graph->input_count = 0;
  graph->model_input_id = NM_IR_INVALID_ID;
  graph->model_output_id = NM_IR_INVALID_ID;
}

uint32_t nm_ir_graph_add_value(NmIrGraph *graph, const char *name,
                               NmIrShape shape, NmIrValueKind kind) {
  if (!graph || graph->value_count >= NM_IR_INVALID_ID)
    return NM_IR_INVALID_ID;
  if (graph->value_count == graph->value_capacity) {
    size_t capacity = graph->value_capacity ? graph->value_capacity * 2 : 16;
    NmIrValue *values = realloc(graph->values, capacity * sizeof(*values));
    if (!values)
      return NM_IR_INVALID_ID;
    graph->values = values;
    graph->value_capacity = capacity;
  }
  NmIrValue *value = &graph->values[graph->value_count];
  memset(value, 0, sizeof(*value));
  snprintf(value->name, sizeof(value->name), "%s", name ? name : "");
  value->shape = shape;
  value->kind = (uint8_t)kind;
  value->producer_node_id = NM_IR_INVALID_ID;
  return (uint32_t)graph->value_count++;
}

uint32_t nm_ir_graph_add_node(NmIrGraph *graph, const NmIrNode *node) {
  if (!graph || !node || graph->node_count >= NM_IR_INVALID_ID)
    return NM_IR_INVALID_ID;
  if (graph->node_count == graph->node_capacity) {
    size_t capacity = graph->node_capacity ? graph->node_capacity * 2 : 16;
    NmIrNode *nodes = realloc(graph->nodes, capacity * sizeof(*nodes));
    if (!nodes)
      return NM_IR_INVALID_ID;
    graph->nodes = nodes;
    graph->node_capacity = capacity;
  }
  graph->nodes[graph->node_count] = *node;
  return (uint32_t)graph->node_count++;
}

bool nm_ir_graph_clone_owned(NmIrGraph *dst, const NmIrGraph *src) {
  if (!dst || !src || dst->value_count != 0 || dst->node_count != 0)
    return false;
  for (size_t i = 0; i < src->value_count; ++i) {
    const NmIrValue *from = &src->values[i];
    uint32_t id = nm_ir_graph_add_value(dst, from->name, from->shape,
                                        (NmIrValueKind)from->kind);
    if (id == NM_IR_INVALID_ID)
      goto fail;
    NmIrValue *to = &dst->values[id];
    to->producer_node_id = from->producer_node_id;
    to->dtype = from->dtype;
    to->declared_batch = from->declared_batch;
    if (from->kind != NM_IR_VALUE_CONSTANT)
      continue;
    if (from->raw) {
      size_t bytes = from->data_count * (from->dtype == NM_IR_DTYPE_I64 ? sizeof(int64_t) : 1);
      to->raw = malloc(bytes ? bytes : 1);
      if (!to->raw)
        goto fail;
      memcpy(to->raw, from->raw, bytes);
      to->data_count = from->data_count;
    } else if (from->tensor) {
      to->tensor = st_clone(from->tensor);
      if (!to->tensor)
        goto fail;
      to->owns_data = true;
      to->data = to->tensor->dtype == ST_DTYPE_F32 ? to->tensor->values : NULL;
      to->data_count = from->data_count;
    } else if (from->data) {
      to->data = malloc(from->data_count * sizeof(float));
      if (!to->data)
        goto fail;
      memcpy(to->data, from->data, from->data_count * sizeof(float));
      to->data_count = from->data_count;
      to->owns_data = true;
    }
  }
  for (size_t i = 0; i < src->node_count; ++i) {
    if (nm_ir_graph_add_node(dst, &src->nodes[i]) == NM_IR_INVALID_ID)
      goto fail;
  }
  dst->model_input_id = src->model_input_id;
  memcpy(dst->input_ids, src->input_ids, sizeof(dst->input_ids));
  dst->input_count = src->input_count;
  dst->model_output_id = src->model_output_id;
  return true;

fail:
  nm_ir_graph_reset(dst);
  return false;
}

static const char *const OP_NAMES[] = {
    [NM_IR_OP_INPUT] = "Input",       [NM_IR_OP_CONSTANT] = "Constant",
    [NM_IR_OP_MATMUL] = "MatMul",     [NM_IR_OP_ADD] = "Add",
    [NM_IR_OP_RELU] = "ReLU",         [NM_IR_OP_SIGMOID] = "Sigmoid",
    [NM_IR_OP_TANH] = "Tanh",         [NM_IR_OP_SOFTMAX] = "Softmax",
    [NM_IR_OP_CONV2D] = "Conv2D",     [NM_IR_OP_MAXPOOL2D] = "MaxPool2D",
    [NM_IR_OP_RESHAPE] = "Reshape",   [NM_IR_OP_SUB] = "Sub",
    [NM_IR_OP_MUL] = "Mul",           [NM_IR_OP_DIV] = "Div",
    [NM_IR_OP_TRANSPOSE] = "Transpose", [NM_IR_OP_LAYERNORM] = "LayerNorm",
    [NM_IR_OP_GELU] = "GELU",         [NM_IR_OP_GATHER] = "Gather",
    [NM_IR_OP_EQUAL] = "Equal",       [NM_IR_OP_NOT] = "Not",
    [NM_IR_OP_WHERE] = "Where",       [NM_IR_OP_CAST] = "Cast",
    [NM_IR_OP_EXPAND] = "Expand",     [NM_IR_OP_SLICE] = "Slice",
    [NM_IR_OP_REDUCEMAX] = "ReduceMax", [NM_IR_OP_RANGE] = "Range",
    [NM_IR_OP_SQUEEZE] = "Squeeze",   [NM_IR_OP_UNSQUEEZE] = "Unsqueeze",
};

const char *nm_ir_dtype_name(NmIrDType dtype) {
  switch (dtype) {
  case NM_IR_DTYPE_BF16: return "bfloat16";
  case NM_IR_DTYPE_F16: return "float16";
  case NM_IR_DTYPE_I64: return "int64";
  case NM_IR_DTYPE_BOOL: return "bool";
  default: return "float32";
  }
}

const char *nm_ir_op_name(NmIrOpType op) {
  return (size_t)op < sizeof(OP_NAMES) / sizeof(OP_NAMES[0]) ? OP_NAMES[op]
                                                              : "Unknown";
}

bool nm_ir_op_from_name(const char *name, NmIrOpType *out) {
  for (size_t i = 0; i < sizeof(OP_NAMES) / sizeof(OP_NAMES[0]); ++i) {
    if (OP_NAMES[i] && name && !strcmp(OP_NAMES[i], name)) {
      *out = (NmIrOpType)i;
      return true;
    }
  }
  return false;
}
