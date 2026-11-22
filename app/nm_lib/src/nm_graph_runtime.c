/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/*
 * Layer entries for the steps of an execution plan. The layer kernels used for
 * training and inference work on NmLayerEntry; this builds them from a graph
 * so that a graph without a frontend model can run and train as well.
 */

#include "nm_model_internal.h"
#include <stdlib.h>
#include <string.h>

typedef union {
  struct {
    NmDenseParams params;
    FloatMatrix weights;
    FloatMatrix bias;
  } dense;
  NmConv2dInternalParams conv;
  NmPool2dInternalParams pool;
} StepParams;

static NmShape layer_shape(const NmIrShape *shape) {
  NmShape out = {0};
  if (shape->ndim == 0)
    return out;
  out.ndim = shape->ndim - 1;
  for (size_t i = 1; i < shape->ndim && i <= 3; ++i)
    out.dims[i - 1] = shape->dims[i];
  return out;
}

static NmStatus bind_step(const NmIrGraph *graph, const NmExecPlanStep *step,
                          NmLayerEntry *layer, StepParams *params) {
  const NmIrNode *node = &graph->nodes[step->node_id];
  layer->input_shape = layer_shape(&graph->values[step->input_value].shape);
  layer->output_shape = layer_shape(&graph->values[step->output_value].shape);
  layer->activation = step->activation;

  switch (step->kind) {
  case NM_EXEC_STEP_DENSE: {
    const NmIrValue *weight = &graph->values[step->weight_value];
    const NmIrValue *bias = &graph->values[step->bias_value];
    params->dense.weights = (FloatMatrix){.rows = weight->shape.dims[0],
                                          .cols = weight->shape.dims[1],
                                          .capacity = weight->data_count,
                                          .values = weight->data};
    params->dense.bias = (FloatMatrix){.rows = 1,
                                       .cols = bias->shape.dims[0],
                                       .capacity = bias->data_count,
                                       .values = bias->data};
    params->dense.params.weights = &params->dense.weights;
    params->dense.params.bias = &params->dense.bias;
    layer->type = NM_LAYER_DENSE;
    layer->params = &params->dense.params;
    break;
  }
  case NM_EXEC_STEP_CONV2D:
    params->conv.weight = graph->values[step->weight_value].tensor;
    params->conv.bias = graph->values[step->bias_value].tensor;
    params->conv.conv = st_conv2d_default_params();
    params->conv.conv.stride_h = node->cfg.conv2d.stride_h;
    params->conv.conv.stride_w = node->cfg.conv2d.stride_w;
    params->conv.conv.pad_h = node->cfg.conv2d.pad_h;
    params->conv.conv.pad_w = node->cfg.conv2d.pad_w;
    params->conv.algo = node->cfg.conv2d.algo;
    params->conv.layout = NM_LAYOUT_NCHW;
    layer->type = NM_LAYER_CONV2D;
    layer->params = &params->conv;
    break;
  case NM_EXEC_STEP_MAXPOOL2D:
    params->pool = (NmPool2dInternalParams){node->cfg.maxpool2d.kernel_h,
                                            node->cfg.maxpool2d.kernel_w,
                                            node->cfg.maxpool2d.stride_h,
                                            node->cfg.maxpool2d.stride_w,
                                            node->cfg.maxpool2d.pad_h,
                                            node->cfg.maxpool2d.pad_w};
    layer->type = NM_LAYER_MAXPOOL2D;
    layer->params = &params->pool;
    break;
  case NM_EXEC_STEP_FLATTEN:
    layer->type = NM_LAYER_FLATTEN;
    break;
  case NM_EXEC_STEP_ACTIVATION:
    /* Stand-alone activations have no layer kernel; the executor applies them
     * itself and training rejects plans that contain one. */
    layer->type = NM_LAYER_FLATTEN;
    return NM_STATUS_OK;
  }
  layer->ops = layer_ops_for_type(layer->type);
  return NM_STATUS_OK;
}

NmStatus layer_runtime_build(NmLayerRuntime *runtime, const NmIrGraph *graph,
                             const NmExecPlan *plan) {
  if (!runtime || !graph || !plan)
    return NM_STATUS_INVALID_ARGUMENT;
  memset(runtime, 0, sizeof(*runtime));
  if (plan->count == 0)
    return NM_STATUS_INVALID_ARGUMENT;

  runtime->layers = calloc(plan->count, sizeof(*runtime->layers));
  StepParams *params = calloc(plan->count, sizeof(*params));
  if (!runtime->layers || !params) {
    free(runtime->layers);
    free(params);
    memset(runtime, 0, sizeof(*runtime));
    return NM_STATUS_ALLOC_FAILED;
  }
  runtime->params = params;
  runtime->count = plan->count;
  runtime->owned = true;
  for (size_t i = 0; i < plan->count; ++i) {
    NmStatus status = bind_step(graph, &plan->steps[i], &runtime->layers[i],
                                &params[i]);
    if (status != NM_STATUS_OK) {
      layer_runtime_destroy(runtime);
      return status;
    }
  }
  return NM_STATUS_OK;
}

void layer_runtime_destroy(NmLayerRuntime *runtime) {
  if (!runtime)
    return;
  if (runtime->owned && runtime->layers) {
    for (size_t i = 0; i < runtime->count; ++i) {
      NmLayerEntry *layer = &runtime->layers[i];
      if (layer->type == NM_LAYER_DENSE && layer->params) {
        NmDenseParams *p = layer->params;
        optimizer_slot_free(&p->w_slot);
        optimizer_slot_free(&p->b_slot);
      } else if (layer->type == NM_LAYER_CONV2D && layer->params) {
        NmConv2dInternalParams *p = layer->params;
        optimizer_slot_free(&p->w_slot);
        optimizer_slot_free(&p->b_slot);
      }
    }
    free(runtime->layers);
    free(runtime->params);
  }
  memset(runtime, 0, sizeof(*runtime));
}
