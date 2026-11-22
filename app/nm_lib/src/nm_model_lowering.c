/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/*
 * Lower the sequential nm model definition into the internal graph IR. The
 * result has the same shape as an exported GraphIR document: constants,
 * MatMul/Add/activation chains, Conv2D, MaxPool2D and Reshape. Constants
 * borrow the model's weights, so later training updates stay visible.
 */

#include "nm_model_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static NmIrShape batched_shape(const NmShape *shape) {
  NmIrShape out = {.ndim = shape->ndim + 1};
  out.dims[0] = 0; /* any batch size */
  for (size_t i = 0; i < shape->ndim && i < NM_IR_MAX_DIMS - 1; ++i)
    out.dims[i + 1] = shape->dims[i];
  return out;
}

static uint32_t add_op_value(NmIrGraph *graph, size_t layer, const char *suffix,
                             NmIrShape shape, NmIrOpType op,
                             const uint32_t *inputs, uint32_t input_count) {
  char name[64];
  snprintf(name, sizeof(name), "layer.%zu.%s", layer, suffix);
  uint32_t value = nm_ir_graph_add_value(graph, name, shape,
                                         NM_IR_VALUE_INTERMEDIATE);
  if (value == NM_IR_INVALID_ID)
    return NM_IR_INVALID_ID;
  NmIrNode node;
  memset(&node, 0, sizeof(node));
  node.op_type = (uint8_t)op;
  node.output_id = value;
  node.input_count = input_count;
  for (uint32_t i = 0; i < input_count; ++i)
    node.input_ids[i] = inputs[i];
  uint32_t node_id = nm_ir_graph_add_node(graph, &node);
  if (node_id == NM_IR_INVALID_ID)
    return NM_IR_INVALID_ID;
  graph->values[value].producer_node_id = node_id;
  return value;
}

static uint32_t add_constant(NmIrGraph *graph, size_t layer, const char *suffix,
                             NmIrShape shape, float *data, size_t count,
                             FloatTensor *tensor) {
  char name[64];
  snprintf(name, sizeof(name), "layer.%zu.%s", layer, suffix);
  uint32_t value = nm_ir_graph_add_value(graph, name, shape,
                                         NM_IR_VALUE_CONSTANT);
  if (value == NM_IR_INVALID_ID)
    return NM_IR_INVALID_ID;
  graph->values[value].data = data;
  graph->values[value].data_count = count;
  graph->values[value].tensor = tensor;
  NmIrNode node;
  memset(&node, 0, sizeof(node));
  node.op_type = NM_IR_OP_CONSTANT;
  node.output_id = value;
  uint32_t node_id = nm_ir_graph_add_node(graph, &node);
  if (node_id == NM_IR_INVALID_ID)
    return NM_IR_INVALID_ID;
  graph->values[value].producer_node_id = node_id;
  return value;
}

static NmIrOpType activation_op(NmActivation activation) {
  switch (activation) {
  case NM_ACT_RELU:
    return NM_IR_OP_RELU;
  case NM_ACT_SIGMOID:
    return NM_IR_OP_SIGMOID;
  case NM_ACT_TANH:
    return NM_IR_OP_TANH;
  case NM_ACT_SOFTMAX:
    return NM_IR_OP_SOFTMAX;
  case NM_ACT_NONE:
    break;
  }
  return NM_IR_OP_INPUT; /* unused */
}

/* Returns the id of the value holding the activation result, or `value`. */
static uint32_t add_activation(NmIrGraph *graph, size_t layer,
                               NmActivation activation, uint32_t value) {
  if (activation == NM_ACT_NONE)
    return value;
  return add_op_value(graph, layer, "output", graph->values[value].shape,
                      activation_op(activation), &value, 1);
}

static float *tensor_f32_data(FloatTensor *tensor) {
  return tensor && tensor->dtype == ST_DTYPE_F32 ? tensor->values : NULL;
}

static NmStatus lower_dense(NmIrGraph *graph, const NmLayerEntry *layer,
                            size_t index, uint32_t *current) {
  const NmDenseParams *params = (const NmDenseParams *)layer->params;
  NmIrShape weight_shape = {.ndim = 2,
                            .dims = {params->weights->rows,
                                     params->weights->cols}};
  NmIrShape bias_shape = {.ndim = 1, .dims = {params->bias->cols}};
  uint32_t weight = add_constant(
      graph, index, "weight", weight_shape, params->weights->values,
      params->weights->rows * params->weights->cols, NULL);
  uint32_t bias = add_constant(graph, index, "bias", bias_shape,
                               params->bias->values, params->bias->cols, NULL);
  if (weight == NM_IR_INVALID_ID || bias == NM_IR_INVALID_ID)
    return NM_STATUS_ALLOC_FAILED;
  NmIrShape out_shape = {.ndim = 2, .dims = {0, params->weights->cols}};
  uint32_t matmul_inputs[2] = {*current, weight};
  uint32_t linear = add_op_value(graph, index, "linear", out_shape,
                                 NM_IR_OP_MATMUL, matmul_inputs, 2);
  if (linear == NM_IR_INVALID_ID)
    return NM_STATUS_ALLOC_FAILED;
  uint32_t add_inputs[2] = {linear, bias};
  uint32_t biased = add_op_value(graph, index, "biased", out_shape,
                                 NM_IR_OP_ADD, add_inputs, 2);
  if (biased == NM_IR_INVALID_ID)
    return NM_STATUS_ALLOC_FAILED;
  *current = add_activation(graph, index, layer->activation, biased);
  return *current == NM_IR_INVALID_ID ? NM_STATUS_ALLOC_FAILED : NM_STATUS_OK;
}

static NmStatus lower_conv2d(NmIrGraph *graph, const NmLayerEntry *layer,
                             size_t index, uint32_t *current) {
  NmConv2dInternalParams *params = (NmConv2dInternalParams *)layer->params;
  if (params->layout != NM_LAYOUT_NCHW || params->conv.dilation_h != 1 ||
      params->conv.dilation_w != 1 || params->weight->ndim != 4)
    return NM_STATUS_UNSUPPORTED_TYPE;
  NmIrShape weight_shape = {.ndim = 4};
  for (size_t i = 0; i < 4; ++i)
    weight_shape.dims[i] = params->weight->shape[i];
  NmIrShape bias_shape = {.ndim = 1, .dims = {params->bias->numel}};
  uint32_t weight = add_constant(graph, index, "weight", weight_shape,
                                 tensor_f32_data(params->weight),
                                 params->weight->numel, params->weight);
  uint32_t bias = add_constant(graph, index, "bias", bias_shape,
                               tensor_f32_data(params->bias),
                               params->bias->numel, params->bias);
  if (weight == NM_IR_INVALID_ID || bias == NM_IR_INVALID_ID)
    return NM_STATUS_ALLOC_FAILED;
  uint32_t inputs[3] = {*current, weight, bias};
  uint32_t conv = add_op_value(graph, index, "conv",
                               batched_shape(&layer->output_shape),
                               NM_IR_OP_CONV2D, inputs, 3);
  if (conv == NM_IR_INVALID_ID)
    return NM_STATUS_ALLOC_FAILED;
  NmIrNode *stored = &graph->nodes[graph->values[conv].producer_node_id];
  stored->cfg.conv2d.stride_h = params->conv.stride_h;
  stored->cfg.conv2d.stride_w = params->conv.stride_w;
  stored->cfg.conv2d.pad_h = params->conv.pad_h;
  stored->cfg.conv2d.pad_w = params->conv.pad_w;
  stored->cfg.conv2d.algo = params->algo;
  *current = add_activation(graph, index, layer->activation, conv);
  return *current == NM_IR_INVALID_ID ? NM_STATUS_ALLOC_FAILED : NM_STATUS_OK;
}

static NmStatus lower_maxpool2d(NmIrGraph *graph, const NmLayerEntry *layer,
                                size_t index, uint32_t *current) {
  const NmPool2dInternalParams *params =
      (const NmPool2dInternalParams *)layer->params;
  uint32_t pool = add_op_value(graph, index, "pool",
                               batched_shape(&layer->output_shape),
                               NM_IR_OP_MAXPOOL2D, current, 1);
  if (pool == NM_IR_INVALID_ID)
    return NM_STATUS_ALLOC_FAILED;
  NmIrNode *stored = &graph->nodes[graph->values[pool].producer_node_id];
  stored->cfg.maxpool2d.kernel_h = params->kernel_h;
  stored->cfg.maxpool2d.kernel_w = params->kernel_w;
  stored->cfg.maxpool2d.stride_h = params->stride_h;
  stored->cfg.maxpool2d.stride_w = params->stride_w;
  stored->cfg.maxpool2d.pad_h = params->pad_h;
  stored->cfg.maxpool2d.pad_w = params->pad_w;
  *current = pool;
  return NM_STATUS_OK;
}

static NmStatus lower_flatten(NmIrGraph *graph, const NmLayerEntry *layer,
                              size_t index, uint32_t *current) {
  size_t flat = 1;
  for (size_t i = 0; i < layer->input_shape.ndim; ++i)
    flat *= layer->input_shape.dims[i];
  NmIrShape shape = {.ndim = 2, .dims = {0, flat}};
  uint32_t value = add_op_value(graph, index, "flatten", shape,
                                NM_IR_OP_RESHAPE, current, 1);
  if (value == NM_IR_INVALID_ID)
    return NM_STATUS_ALLOC_FAILED;
  *current = value;
  return NM_STATUS_OK;
}

NmStatus nm_seq_lower_to_ir(NmSeq *m) {
  if (!m || (m->input_shape.ndim != 1 && m->input_shape.ndim != 3))
    return NM_STATUS_INVALID_ARGUMENT;

  NmIrGraph *graph = &m->compiled_graph;
  nm_ir_graph_reset(graph);

  graph->model_input_id = nm_ir_graph_add_value(
      graph, "input", batched_shape(&m->input_shape), NM_IR_VALUE_MODEL_INPUT);
  NmIrNode input_node;
  memset(&input_node, 0, sizeof(input_node));
  input_node.op_type = NM_IR_OP_INPUT;
  input_node.output_id = graph->model_input_id;
  if (graph->model_input_id == NM_IR_INVALID_ID ||
      nm_ir_graph_add_node(graph, &input_node) == NM_IR_INVALID_ID) {
    nm_ir_graph_reset(graph);
    return NM_STATUS_ALLOC_FAILED;
  }
  graph->values[graph->model_input_id].producer_node_id = 0;
  graph->input_ids[0] = graph->model_input_id;
  graph->input_count = 1;

  uint32_t current = graph->model_input_id;
  for (size_t i = 0; i < m->num_layers; ++i) {
    const NmLayerEntry *layer = &m->layers[i];
    NmStatus status;
    switch (layer->type) {
    case NM_LAYER_DENSE:
      status = lower_dense(graph, layer, i, &current);
      break;
    case NM_LAYER_CONV2D:
      status = lower_conv2d(graph, layer, i, &current);
      break;
    case NM_LAYER_MAXPOOL2D:
      status = lower_maxpool2d(graph, layer, i, &current);
      break;
    case NM_LAYER_FLATTEN:
      status = lower_flatten(graph, layer, i, &current);
      break;
    default:
      status = NM_STATUS_UNSUPPORTED_TYPE;
      break;
    }
    if (status != NM_STATUS_OK) {
      nm_ir_graph_reset(graph);
      return status;
    }
  }

  graph->values[current].kind = NM_IR_VALUE_MODEL_OUTPUT;
  graph->model_output_id = current;
  return NM_STATUS_OK;
}
