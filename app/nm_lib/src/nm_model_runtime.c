/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/*
 * Shared runtime helpers for the nm model implementation.
 */

#include "app/nm_lib/src/nm_model_internal.h"
#include "sv_optim.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <omp.h>
#pragma GCC diagnostic pop
#endif

bool model_weights_are_f32(const NmSeq *m) {
  if (!m)
    return true;
  for (size_t i = 0; i < m->num_layers; ++i) {
    const NmLayerEntry *e = &m->layers[i];
    if (e->type == NM_LAYER_CONV2D && e->params) {
      const NmConv2dInternalParams *p =
          (const NmConv2dInternalParams *)e->params;
      if (p->weight && p->weight->dtype != ST_DTYPE_F32)
        return false;
      if (p->bias && p->bias->dtype != ST_DTYPE_F32)
        return false;
    }
  }
  return true;
}

size_t shape_flat_size(NmShape s) {
  size_t n = 1;
  for (size_t i = 0; i < s.ndim; ++i)
    n *= s.dims[i];
  return n;
}

NmShape shape_from_tensor(const FloatTensor *t) {
  NmShape s = {0};
  if (!t)
    return s;
  if (t->ndim == 2) {
    s.ndim = 1;
    s.dims[0] = t->shape[1];
    return s;
  }
  if (t->ndim == 4) {
    s.ndim = 3;
    s.dims[0] = t->shape[1];
    s.dims[1] = t->shape[2];
    s.dims[2] = t->shape[3];
  }
  return s;
}

NmShape model_tail_shape(const NmSeq *m) {
  return m->num_layers > 0 ? m->layers[m->num_layers - 1].output_shape
                           : m->input_shape;
}

NmStatus ensure_capacity(NmSeq *m) {
  if (m->num_layers < m->capacity)
    return NM_STATUS_OK;
  size_t new_cap = m->capacity * 2;
  NmLayerEntry *tmp = realloc(m->layers, new_cap * sizeof(NmLayerEntry));
  if (!tmp)
    return NM_STATUS_ALLOC_FAILED;
  m->layers = tmp;
  m->capacity = new_cap;
  return NM_STATUS_OK;
}

bool compile_uses_softmax_ce_shortcut(const NmSeq *m, NmLoss loss_type) {
  if (!m || m->num_layers == 0 || loss_type != NM_LOSS_CROSS_ENTROPY)
    return false;
  return m->layers[m->num_layers - 1].activation == NM_ACT_SOFTMAX;
}

NmOptimizerConfig optimizer_defaults_internal(void) {
  NmOptimizerConfig cfg;
  cfg.type = NM_OPT_ADAM;
  cfg.learning_rate = 1e-3f;
  cfg.momentum = 0.9f;
  cfg.beta1 = 0.9f;
  cfg.beta2 = 0.999f;
  cfg.epsilon = 1e-8f;
  return cfg;
}

bool optimizer_config_valid(NmOptimizerConfig cfg) {
  if (cfg.type < NM_OPT_SGD || cfg.type > NM_OPT_ADAM)
    return false;
  if (!(cfg.learning_rate > 0.0f))
    return false;
  if (!(cfg.epsilon > 0.0f))
    return false;
  if (cfg.type == NM_OPT_MOMENTUM &&
      (cfg.momentum < 0.0f || cfg.momentum >= 1.0f)) {
    return false;
  }
  if ((cfg.type == NM_OPT_ADAM || cfg.type == NM_OPT_RMSPROP) &&
      (cfg.beta2 <= 0.0f || cfg.beta2 >= 1.0f)) {
    return false;
  }
  if (cfg.type == NM_OPT_ADAM && (cfg.beta1 <= 0.0f || cfg.beta1 >= 1.0f))
    return false;
  return true;
}

void optimizer_slot_free(NmOptSlot *slot) {
  if (!slot)
    return;
  free(slot->m);
  free(slot->v);
  slot->m = NULL;
  slot->v = NULL;
  slot->size = 0;
}

bool optimizer_slot_ensure(NmOptSlot *slot, size_t size, bool need_m,
                           bool need_v) {
  if (!slot)
    return false;
  if (slot->size != size) {
    optimizer_slot_free(slot);
    slot->size = size;
  }
  if (need_m && !slot->m) {
    slot->m = (float *)calloc(size, sizeof(float));
    if (!slot->m)
      return false;
  }
  if (need_v && !slot->v) {
    slot->v = (float *)calloc(size, sizeof(float));
    if (!slot->v)
      return false;
  }
  return true;
}

uint64_t align_u64(uint64_t value, uint32_t alignment) {
  uint64_t a = alignment > 0 ? (uint64_t)alignment : 1ULL;
  return value + ((a - (value % a)) % a);
}

const NmBackendOps *model_backend_ops(const NmSeq *m) {
  if (!m || !m->runtime.backend_ops)
    return &NM_BACKEND_DEFAULT_OPS;
  return m->runtime.backend_ops;
}

const NmBackendCapabilities *model_backend_caps(const NmSeq *m) {
  if (!m || !m->runtime.backend_caps)
    return nm_backend_capabilities();
  return m->runtime.backend_caps;
}

void exec_plan_init(NmExecPlan *plan) {
  if (!plan)
    return;
  memset(plan, 0, sizeof(*plan));
  plan->input_value_id = NM_IR_INVALID_ID;
  plan->model_output_id = NM_IR_INVALID_ID;
}

void exec_plan_reset(NmExecPlan *plan) {
  if (!plan)
    return;
  NmExecPlanStep *steps = plan->steps;
  size_t capacity = plan->capacity;
  exec_plan_init(plan);
  plan->steps = steps;
  plan->capacity = capacity;
}

void exec_plan_destroy(NmExecPlan *plan) {
  if (!plan)
    return;
  free(plan->steps);
  exec_plan_init(plan);
}

static bool exec_plan_reserve(NmExecPlan *plan, size_t count) {
  if (!plan)
    return false;
  if (count <= plan->capacity)
    return true;

  size_t new_cap = plan->capacity == 0 ? 8 : plan->capacity;
  while (new_cap < count)
    new_cap *= 2;

  NmExecPlanStep *steps = realloc(plan->steps, new_cap * sizeof(*steps));
  if (!steps)
    return false;
  plan->steps = steps;
  plan->capacity = new_cap;
  return true;
}

static bool activation_from_op(uint8_t op, NmActivation *out) {
  switch (op) {
  case NM_IR_OP_RELU:
    *out = NM_ACT_RELU;
    return true;
  case NM_IR_OP_SIGMOID:
    *out = NM_ACT_SIGMOID;
    return true;
  case NM_IR_OP_TANH:
    *out = NM_ACT_TANH;
    return true;
  case NM_IR_OP_SOFTMAX:
    *out = NM_ACT_SOFTMAX;
    return true;
  default:
    return false;
  }
}

typedef struct {
  const NmIrGraph *graph;
  size_t *uses;
  bool *consumed;
  bool *ready;
} PlanBuild;

/* Finds the node that reads `value` as its first input, or SIZE_MAX. */
static size_t first_reader(const PlanBuild *build, uint32_t value) {
  for (size_t i = 0; i < build->graph->node_count; ++i) {
    const NmIrNode *node = &build->graph->nodes[i];
    if (!build->consumed[i] && node->op_type != NM_IR_OP_CONSTANT &&
        node->input_count > 0 && node->input_ids[0] == value)
      return i;
  }
  return SIZE_MAX;
}

static bool constant_has_data(const NmIrValue *value, size_t ndim) {
  return value->kind == NM_IR_VALUE_CONSTANT && value->shape.ndim == ndim &&
         (value->data || value->tensor);
}

/* Fuses an optional activation that is the only reader of the step output. */
static void fuse_activation(PlanBuild *build, NmExecPlanStep *step) {
  size_t reader = first_reader(build, step->output_value);
  NmActivation activation;
  if (reader == SIZE_MAX || build->uses[step->output_value] != 1 ||
      !activation_from_op(build->graph->nodes[reader].op_type, &activation))
    return;
  build->consumed[reader] = true;
  step->activation = activation;
  step->output_value = build->graph->nodes[reader].output_id;
}

static NmStatus plan_dense(PlanBuild *build, const NmIrNode *matmul,
                           NmExecPlanStep *step) {
  const NmIrGraph *graph = build->graph;
  if (matmul->input_count != 2 ||
      !constant_has_data(&graph->values[matmul->input_ids[1]], 2) ||
      !graph->values[matmul->input_ids[1]].data)
    return NM_STATUS_UNSUPPORTED_TYPE;
  size_t reader = first_reader(build, matmul->output_id);
  if (reader == SIZE_MAX || build->uses[matmul->output_id] != 1)
    return NM_STATUS_UNSUPPORTED_TYPE;
  const NmIrNode *add = &graph->nodes[reader];
  const NmIrValue *weight = &graph->values[matmul->input_ids[1]];
  if (add->op_type != NM_IR_OP_ADD || add->input_count != 2)
    return NM_STATUS_UNSUPPORTED_TYPE;
  const NmIrValue *bias = &graph->values[add->input_ids[1]];
  if (!constant_has_data(bias, 1) || !bias->data ||
      bias->shape.dims[0] != weight->shape.dims[1])
    return NM_STATUS_UNSUPPORTED_TYPE;
  build->consumed[reader] = true;
  step->kind = NM_EXEC_STEP_DENSE;
  step->weight_value = matmul->input_ids[1];
  step->bias_value = add->input_ids[1];
  step->output_value = add->output_id;
  fuse_activation(build, step);
  return NM_STATUS_OK;
}

static NmStatus plan_conv2d(PlanBuild *build, const NmIrNode *conv,
                            NmExecPlanStep *step) {
  const NmIrGraph *graph = build->graph;
  if (conv->input_count != 3 ||
      !constant_has_data(&graph->values[conv->input_ids[1]], 4) ||
      !constant_has_data(&graph->values[conv->input_ids[2]], 1) ||
      !graph->values[conv->input_ids[1]].tensor ||
      !graph->values[conv->input_ids[2]].tensor)
    return NM_STATUS_UNSUPPORTED_TYPE;
  step->kind = NM_EXEC_STEP_CONV2D;
  step->weight_value = conv->input_ids[1];
  step->bias_value = conv->input_ids[2];
  fuse_activation(build, step);
  return NM_STATUS_OK;
}

static NmStatus plan_node(PlanBuild *build, size_t index,
                          NmExecPlanStep *step) {
  const NmIrNode *node = &build->graph->nodes[index];
  step->node_id = (uint32_t)index;
  step->input_value = node->input_ids[0];
  step->output_value = node->output_id;
  step->weight_value = NM_IR_INVALID_ID;
  step->bias_value = NM_IR_INVALID_ID;
  step->activation = NM_ACT_NONE;
  NmActivation activation;
  switch (node->op_type) {
  case NM_IR_OP_MATMUL:
    return plan_dense(build, node, step);
  case NM_IR_OP_CONV2D:
    return plan_conv2d(build, node, step);
  case NM_IR_OP_MAXPOOL2D:
    if (node->input_count != 1)
      return NM_STATUS_UNSUPPORTED_TYPE;
    step->kind = NM_EXEC_STEP_MAXPOOL2D;
    return NM_STATUS_OK;
  case NM_IR_OP_RESHAPE:
    if (node->input_count != 1 ||
        build->graph->values[node->output_id].shape.ndim != 2)
      return NM_STATUS_UNSUPPORTED_TYPE;
    step->kind = NM_EXEC_STEP_FLATTEN;
    return NM_STATUS_OK;
  default:
    if (node->input_count == 1 && activation_from_op(node->op_type, &activation)) {
      step->kind = NM_EXEC_STEP_ACTIVATION;
      step->activation = activation;
      return NM_STATUS_OK;
    }
    return NM_STATUS_UNSUPPORTED_TYPE;
  }
}

static size_t shape_features(const NmIrShape *shape) {
  size_t features = 1;
  for (size_t i = 1; i < shape->ndim; ++i)
    features *= shape->dims[i];
  return features;
}

static NmStatus plan_endpoints(NmExecPlan *plan, const NmIrGraph *graph) {
  if (graph->model_input_id >= graph->value_count ||
      graph->model_output_id >= graph->value_count)
    return NM_STATUS_INVALID_ARGUMENT;
  const NmIrShape *input = &graph->values[graph->model_input_id].shape;
  const NmIrShape *output = &graph->values[graph->model_output_id].shape;
  if ((input->ndim != 2 && input->ndim != 4) || output->ndim != 2 ||
      output->dims[1] == 0)
    return NM_STATUS_UNSUPPORTED_TYPE;
  plan->input_value_id = graph->model_input_id;
  plan->model_output_id = graph->model_output_id;
  plan->input_shape.ndim = input->ndim - 1;
  for (size_t i = 1; i < input->ndim; ++i)
    plan->input_shape.dims[i - 1] = input->dims[i];
  plan->input_cols = shape_features(input);
  plan->output_cols = output->dims[1];
  return NM_STATUS_OK;
}

NmStatus exec_plan_build(NmExecPlan *plan, const NmIrGraph *graph) {
  if (!plan || !graph)
    return NM_STATUS_INVALID_ARGUMENT;
  exec_plan_reset(plan);

  NmStatus status = plan_endpoints(plan, graph);
  if (status != NM_STATUS_OK) {
    exec_plan_reset(plan);
    return status;
  }
  if (!exec_plan_reserve(plan, graph->node_count + 1)) {
    exec_plan_reset(plan);
    return NM_STATUS_ALLOC_FAILED;
  }

  PlanBuild build = {
      .graph = graph,
      .uses = calloc(graph->value_count, sizeof(size_t)),
      .consumed = calloc(graph->node_count + 1, sizeof(bool)),
      .ready = calloc(graph->value_count, sizeof(bool)),
  };
  if (!build.uses || !build.consumed || !build.ready) {
    status = NM_STATUS_ALLOC_FAILED;
    goto done;
  }
  for (size_t i = 0; i < graph->node_count; ++i) {
    for (uint32_t j = 0; j < graph->nodes[i].input_count; ++j) {
      if (graph->nodes[i].input_ids[j] >= graph->value_count) {
        status = NM_STATUS_FORMAT_ERROR;
        goto done;
      }
      build.uses[graph->nodes[i].input_ids[j]]++;
    }
  }
  build.uses[graph->model_output_id]++;
  build.ready[graph->model_input_id] = true;
  for (size_t i = 0; i < graph->value_count; ++i) {
    if (graph->values[i].kind == NM_IR_VALUE_CONSTANT)
      build.ready[i] = true;
  }

  bool dense_only = true;
  for (size_t i = 0; i < graph->node_count && status == NM_STATUS_OK; ++i) {
    const NmIrNode *node = &graph->nodes[i];
    if (build.consumed[i] || node->op_type == NM_IR_OP_INPUT ||
        node->op_type == NM_IR_OP_CONSTANT)
      continue;
    build.consumed[i] = true;
    NmExecPlanStep step;
    memset(&step, 0, sizeof(step));
    status = plan_node(&build, i, &step);
    if (status != NM_STATUS_OK)
      break;
    if (!build.ready[step.input_value]) {
      status = NM_STATUS_FORMAT_ERROR;
      break;
    }
    build.ready[step.output_value] = true;
    dense_only = dense_only && step.kind == NM_EXEC_STEP_DENSE;
    plan->steps[plan->count++] = step;
  }
  if (status == NM_STATUS_OK &&
      (plan->count == 0 || !build.ready[graph->model_output_id] ||
       plan->steps[plan->count - 1].output_value != graph->model_output_id))
    status = NM_STATUS_FORMAT_ERROR;
  plan->dense_only = dense_only && plan->input_shape.ndim == 1;

done:
  free(build.uses);
  free(build.consumed);
  free(build.ready);
  if (status != NM_STATUS_OK)
    exec_plan_reset(plan);
  return status;
}

bool inference_plan_init(NmInferencePlan *plan) {
  if (!plan)
    return false;
  memset(plan, 0, sizeof(*plan));
  return tensor_pool_init(&plan->pool);
}

void inference_plan_reset(NmInferencePlan *plan) {
  if (!plan)
    return;
  tensor_pool_reset(&plan->pool);
}

void inference_plan_destroy(NmInferencePlan *plan) {
  if (!plan)
    return;
  tensor_pool_destroy(&plan->pool);
  plan->prepared_batch_rows = 0;
  plan->output_cols = 0;
  plan->peak_tensor_count = 0;
  plan->peak_tensor_bytes = 0;
  plan->prepared = false;
}

bool inference_plan_reinit(NmInferencePlan *plan) {
  if (!plan)
    return false;
  inference_plan_destroy(plan);
  return inference_plan_init(plan);
}

static size_t tensor_pool_bytes(const NmTensorPool *pool) {
  size_t total = 0;
  if (!pool)
    return 0;
  for (size_t i = 0; i < pool->count; ++i) {
    const FloatTensor *t = pool->entries[i].tensor;
    if (!t)
      continue;
    total += t->capacity * sizeof(float);
  }
  return total;
}

void inference_plan_capture(NmInferencePlan *plan, const NmTensorPool *pool,
                            size_t batch_rows, size_t output_cols) {
  if (!plan || !pool)
    return;
  plan->prepared_batch_rows = batch_rows;
  plan->output_cols = output_cols;
  if (pool->count > plan->peak_tensor_count)
    plan->peak_tensor_count = pool->count;
  size_t pool_bytes = tensor_pool_bytes(pool);
  if (pool_bytes > plan->peak_tensor_bytes)
    plan->peak_tensor_bytes = pool_bytes;
  plan->prepared = true;
}

bool tensor_init_borrowed_view(FloatTensor *view, size_t ndim,
                               const size_t *shape, float *data,
                               size_t capacity, StLayout layout) {
  if (!view || !shape || !data)
    return false;
  memset(view, 0, sizeof(*view));
  if (!st_numel_from_shape(ndim, shape, &view->numel) ||
      !st_compute_default_strides(ndim, shape, view->strides) ||
      view->numel > capacity) {
    return false;
  }
  view->ndim = ndim;
  memcpy(view->shape, shape, ndim * sizeof(size_t));
  view->capacity = capacity;
  view->values = data;
  view->owns_data = false;
  view->layout = layout;
  view->buf = NULL;
  view->view_offset = 0;
  view->view_src = NULL;
  view->extra = NULL;
  view->extra_free = NULL;
  return true;
}

bool tensor_pool_init(NmTensorPool *pool) {
  if (!pool)
    return false;
  pool->entries = NULL;
  pool->count = 0;
  pool->capacity = 0;
  return true;
}

void tensor_pool_reset(NmTensorPool *pool) {
  if (!pool)
    return;
  for (size_t i = 0; i < pool->count; ++i)
    pool->entries[i].in_use = false;
}

void tensor_pool_destroy(NmTensorPool *pool) {
  if (!pool)
    return;
  for (size_t i = 0; i < pool->count; ++i)
    st_destroy(pool->entries[i].tensor);
  free(pool->entries);
  pool->entries = NULL;
  pool->count = 0;
  pool->capacity = 0;
}

static bool tensor_same_shape(const FloatTensor *t, size_t ndim,
                              const size_t *shape) {
  if (!t || t->ndim != ndim)
    return false;
  for (size_t i = 0; i < ndim; ++i) {
    if (t->shape[i] != shape[i])
      return false;
  }
  return true;
}

FloatTensor *tensor_pool_acquire(NmTensorPool *pool, size_t ndim,
                                 const size_t *shape) {
  if (!pool || !shape)
    return NULL;

  for (size_t i = 0; i < pool->count; ++i) {
    NmTensorPoolEntry *entry = &pool->entries[i];
    if (!entry->in_use && tensor_same_shape(entry->tensor, ndim, shape)) {
      entry->in_use = true;
      return entry->tensor;
    }
  }

  FloatTensor *tensor = st_create(ndim, (size_t *)shape);
  if (!tensor)
    return NULL;

  if (pool->count == pool->capacity) {
    size_t new_cap = pool->capacity == 0 ? 16 : pool->capacity * 2;
    NmTensorPoolEntry *new_entries = (NmTensorPoolEntry *)realloc(
        pool->entries, new_cap * sizeof(*new_entries));
    if (!new_entries) {
      st_destroy(tensor);
      return NULL;
    }
    pool->entries = new_entries;
    pool->capacity = new_cap;
  }

  pool->entries[pool->count].tensor = tensor;
  pool->entries[pool->count].in_use = true;
  pool->count += 1;
  return tensor;
}

FloatTensor *tensor_pool_clone(NmTensorPool *pool, const FloatTensor *src) {
  if (!pool || !src)
    return NULL;
  FloatTensor *dst = tensor_pool_acquire(pool, src->ndim, src->shape);
  if (!dst)
    return NULL;
  memcpy(dst->values, src->values, src->numel * sizeof(float));
  return dst;
}

void exec_trace_init(NmExecTrace *t) {
  if (!t)
    return;
  t->nodes = NULL;
  t->count = 0;
  t->capacity = 0;
}

void exec_trace_reset(NmExecTrace *t) {
  if (!t)
    return;
  t->count = 0;
}

void exec_trace_destroy(NmExecTrace *t) {
  if (!t)
    return;
  free(t->nodes);
  t->nodes = NULL;
  t->count = 0;
  t->capacity = 0;
}

bool exec_trace_push(NmExecTrace *t, NmLayerType layer_type,
                     NmShape output_shape, size_t tensor_numel) {
  if (!t)
    return false;
  if (t->count == t->capacity) {
    size_t new_cap = t->capacity == 0 ? 16 : t->capacity * 2;
    NmExecTraceNode *nodes =
        (NmExecTraceNode *)realloc(t->nodes, new_cap * sizeof(*nodes));
    if (!nodes)
      return false;
    t->nodes = nodes;
    t->capacity = new_cap;
  }
  t->nodes[t->count].layer_type = layer_type;
  t->nodes[t->count].output_shape = output_shape;
  t->nodes[t->count].tensor_numel = tensor_numel;
  t->count += 1;
  return true;
}

bool optimizer_apply_array(const NmOptimizerConfig *cfg, uint64_t step,
                           NmOptSlot *slot, float *params, const float *grads,
                           size_t n, float grad_scale) {
  if (!cfg || !params || !grads || n == 0)
    return false;

  bool need_m = (cfg->type == NM_OPT_MOMENTUM || cfg->type == NM_OPT_ADAM);
  bool need_v = (cfg->type == NM_OPT_RMSPROP || cfg->type == NM_OPT_ADAM);
  if ((need_m || need_v) && !optimizer_slot_ensure(slot, n, need_m, need_v))
    return false;

  const float lr = cfg->learning_rate;
  switch (cfg->type) {
  case NM_OPT_SGD:
    return sv_optim_sgd(params, grads, n, lr, grad_scale, 0.0f);
  case NM_OPT_MOMENTUM:
    return sv_optim_momentum(params, grads, slot->m, n, lr, cfg->momentum,
                             grad_scale, 0.0f);
  case NM_OPT_RMSPROP:
    return sv_optim_rmsprop(params, grads, slot->v, n, lr, cfg->beta2,
                            cfg->epsilon, grad_scale, 0.0f);
  case NM_OPT_ADAM:
    return sv_optim_adam(params, grads, slot->m, slot->v, n, lr, cfg->beta1,
                         cfg->beta2, cfg->epsilon, step, grad_scale, 0.0f);
  }
  return false;
}

bool optimizer_apply_matrix(const NmOptimizerConfig *cfg, uint64_t step,
                            NmOptSlot *slot, FloatMatrix *param,
                            const FloatMatrix *grad, float grad_scale) {
  if (!param || !grad || param->rows != grad->rows || param->cols != grad->cols)
    return false;
  return optimizer_apply_array(cfg, step, slot, param->values, grad->values,
                               param->rows * param->cols, grad_scale);
}

bool optimizer_apply_tensor(const NmOptimizerConfig *cfg, uint64_t step,
                            NmOptSlot *slot, FloatTensor *param,
                            const FloatTensor *grad, float grad_scale) {
  if (!param || !grad || param->numel != grad->numel)
    return false;
  return optimizer_apply_array(cfg, step, slot, param->values, grad->values,
                               param->numel, grad_scale);
}
