/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/*
 * Core model lifecycle, builder and compile support for nm models.
 */

#include "nm_model_serialization.h"
#include <log.h>
#include <stdio.h>
#include <stdlib.h>

NmSeq *nm_seq_create(NmShape input_shape) {
  if (input_shape.ndim != 1 && input_shape.ndim != 3)
    return NULL;
  for (size_t i = 0; i < input_shape.ndim; ++i) {
    if (input_shape.dims[i] == 0)
      return NULL;
  }

  NmSeq *m = calloc(1, sizeof(*m));
  if (!m)
    return NULL;

  m->input_shape = input_shape;
  m->capacity = 8;
  m->layers = calloc(m->capacity, sizeof(NmLayerEntry));
  if (!m->layers) {
    free(m);
    return NULL;
  }
  m->runtime.backend_ops = &NM_BACKEND_DEFAULT_OPS;
  m->runtime.backend_caps = nm_backend_capabilities();
  m->runtime.backend_name = nm_backend();
  m->runtime.tensor_layout = NM_LAYOUT_NCHW;
  exec_plan_init(&m->runtime.exec_plan);
  if (!inference_plan_init(&m->runtime.inference_plan)) {
    free(m->layers);
    free(m);
    return NULL;
  }
  m->training.optimizer_config = optimizer_defaults_internal();
  m->training.optimizer_step = 0;
  m->training.autograd_mode = NM_AUTOGRAD_DISABLED;
  m->training.batch_parallel.enabled = false;
  m->training.batch_parallel.deterministic = true;
  m->training.batch_parallel.num_workers = 0;
  m->training.inference_only = false;
  if (!nm_ir_graph_init(&m->compiled_graph)) {
    inference_plan_destroy(&m->runtime.inference_plan);
    free(m->layers);
    free(m);
    return NULL;
  }
  exec_trace_init(&m->training.last_trace);
  m->storage.format_version = NM_SEQ_BIN_VERSION;
  m->storage.alignment = NM_SEQ_ALIGNMENT;
  m->storage.inference_only = false;
  return m;
}

NmOptimizerConfig nm_seq_optimizer_defaults(void) {
  return optimizer_defaults_internal();
}

NmStatus nm_seq_set_optimizer(NmSeq *m, const NmOptimizerConfig *cfg) {
  if (!m)
    return NM_STATUS_INVALID_ARGUMENT;
  if (m->training.inference_only)
    return NM_STATUS_INVALID_ARGUMENT;

  NmOptimizerConfig merged = cfg ? *cfg : optimizer_defaults_internal();
  NmOptimizerConfig def = optimizer_defaults_internal();
  if (!(merged.learning_rate > 0.0f))
    merged.learning_rate = def.learning_rate;
  if (!(merged.epsilon > 0.0f))
    merged.epsilon = def.epsilon;
  if (!(merged.momentum > 0.0f))
    merged.momentum = def.momentum;
  if (!(merged.beta1 > 0.0f))
    merged.beta1 = def.beta1;
  if (!(merged.beta2 > 0.0f))
    merged.beta2 = def.beta2;

  if (!optimizer_config_valid(merged))
    return NM_STATUS_INVALID_ARGUMENT;

  m->training.optimizer_config = merged;
  m->training.optimizer_step = 0;
  return NM_STATUS_OK;
}

NmStatus nm_seq_set_backend(NmSeq *m, NmBackendType backend) {
  if (!m)
    return NM_STATUS_INVALID_ARGUMENT;

  switch (backend) {
  case NM_BACKEND_AUTO:
  case NM_BACKEND_DEFAULT:
    m->runtime.backend_ops = &NM_BACKEND_DEFAULT_OPS;
    m->runtime.backend_caps = nm_backend_capabilities();
    m->runtime.backend_name = nm_backend();
    if (!inference_plan_reinit(&m->runtime.inference_plan))
      return NM_STATUS_ALLOC_FAILED;
    return NM_STATUS_OK;
  default:
    return NM_STATUS_UNSUPPORTED_TYPE;
  }
}

const char *nm_seq_backend(const NmSeq *m) {
  if (!m || !m->runtime.backend_name)
    return "unknown";
  return m->runtime.backend_name;
}

const NmBackendCapabilities *nm_seq_model_backend_capabilities(const NmSeq *m) {
  return model_backend_caps(m);
}

bool nm_seq_is_inference_only(const NmSeq *m) {
  return m ? m->training.inference_only : false;
}

NmStatus nm_seq_set_tensor_layout(NmSeq *m, NmTensorLayout layout) {
  if (!m)
    return NM_STATUS_INVALID_ARGUMENT;
  if (layout != NM_LAYOUT_NCHW && layout != NM_LAYOUT_NHWC)
    return NM_STATUS_INVALID_ARGUMENT;
  m->runtime.tensor_layout = layout;
  if (!inference_plan_reinit(&m->runtime.inference_plan))
    return NM_STATUS_ALLOC_FAILED;
  return NM_STATUS_OK;
}

NmTensorLayout nm_seq_tensor_layout(const NmSeq *m) {
  if (!m)
    return NM_LAYOUT_NCHW;
  return m->runtime.tensor_layout;
}

NmStatus nm_seq_set_autograd_mode(NmSeq *m, NmAutogradMode mode) {
  if (!m)
    return NM_STATUS_INVALID_ARGUMENT;
  if (m->training.inference_only)
    return NM_STATUS_INVALID_ARGUMENT;
  if (mode != NM_AUTOGRAD_DISABLED && mode != NM_AUTOGRAD_TAPE)
    return NM_STATUS_INVALID_ARGUMENT;
  m->training.autograd_mode = mode;
  exec_trace_reset(&m->training.last_trace);
  return NM_STATUS_OK;
}

NmAutogradMode nm_seq_autograd_mode(const NmSeq *m) {
  if (!m)
    return NM_AUTOGRAD_DISABLED;
  return m->training.autograd_mode;
}

NmStatus nm_seq_set_batch_parallel(NmSeq *m, const NmBatchParallelConfig *cfg) {
  if (!m || !cfg)
    return NM_STATUS_INVALID_ARGUMENT;
  if (m->training.inference_only)
    return NM_STATUS_INVALID_ARGUMENT;

  NmBatchParallelConfig merged = *cfg;
  if (!merged.enabled) {
    merged.deterministic = true;
    merged.num_workers = 0;
  }

  m->training.batch_parallel = merged;
  return NM_STATUS_OK;
}

NmBatchParallelConfig nm_seq_batch_parallel(const NmSeq *m) {
  NmBatchParallelConfig cfg = {0};
  if (!m)
    return cfg;
  return m->training.batch_parallel;
}

size_t nm_seq_last_graph_nodes(const NmSeq *m) {
  if (!m)
    return 0;
  return m->training.last_trace.count;
}

void nm_seq_free(NmSeq *m) {
  if (!m)
    return;
  for (size_t i = 0; i < m->num_layers; ++i)
    destroy_layer_params(&m->layers[i]);
  exec_plan_destroy(&m->runtime.exec_plan);
  inference_plan_destroy(&m->runtime.inference_plan);
  exec_trace_destroy(&m->training.last_trace);
  nm_ir_graph_destroy(&m->compiled_graph);
  free(m->layers);
  free(m);
}

NmStatus nm_seq_add_dense(NmSeq *m, const NmDenseConfig *cfg) {
  if (!m || !cfg || cfg->units == 0)
    return NM_STATUS_INVALID_ARGUMENT;
  if (m->compiled)
    return NM_STATUS_INVALID_ARGUMENT;

  NmShape cur = model_tail_shape(m);
  if (cur.ndim != 1) {
    log_error("Dense layer requires flat input (ndim==1). Add "
              "nm_seq_add_flatten() first.");
    return NM_STATUS_INVALID_ARGUMENT;
  }

  NmStatus status = ensure_capacity(m);
  if (status != NM_STATUS_OK)
    return status;

  size_t in_dim = cur.dims[0];
  NmDenseParams *p = calloc(1, sizeof(*p));
  if (!p)
    return NM_STATUS_ALLOC_FAILED;

  p->weights = sm_create_random_he(in_dim, cfg->units, in_dim);
  p->bias = sm_create_zeros(1, cfg->units);
  if (!p->weights || !p->bias) {
    sm_destroy(p->weights);
    sm_destroy(p->bias);
    free(p);
    return NM_STATUS_ALLOC_FAILED;
  }

  NmLayerEntry *e = &m->layers[m->num_layers++];
  e->type = NM_LAYER_DENSE;
  e->input_shape = cur;
  e->output_shape = (NmShape){.ndim = 1, .dims = {cfg->units}};
  e->activation = cfg->activation;
  e->ops = layer_ops_for_type(e->type);
  e->params = p;
  return NM_STATUS_OK;
}

NmStatus nm_seq_add_conv2d(NmSeq *m, const NmConv2dConfig *cfg) {
  if (!m || !cfg || cfg->filters == 0 || cfg->kernel_size == 0)
    return NM_STATUS_INVALID_ARGUMENT;
  if (m->compiled)
    return NM_STATUS_INVALID_ARGUMENT;

  NmShape cur = model_tail_shape(m);
  if (cur.ndim != 3) {
    log_error("Conv2D requires spatial input (ndim==3, C×H×W).");
    return NM_STATUS_INVALID_ARGUMENT;
  }

  NmStatus status = ensure_capacity(m);
  if (status != NM_STATUS_OK)
    return status;

  size_t cin = cur.dims[0];
  size_t h = cur.dims[1];
  size_t w = cur.dims[2];
  size_t kh = cfg->kernel_size;
  size_t kw = cfg->kernel_size;
  size_t stride = cfg->stride ? cfg->stride : 1;
  size_t padding = cfg->padding;

  StConv2dParams conv = st_conv2d_default_params();
  conv.stride_h = stride;
  conv.stride_w = stride;
  conv.pad_h = padding;
  conv.pad_w = padding;

  size_t out_h = 0;
  size_t out_w = 0;
  if (!st_conv2d_output_hw(h, w, kh, kw, &conv, &out_h, &out_w))
    return NM_STATUS_INVALID_ARGUMENT;

  NmConv2dInternalParams *p = calloc(1, sizeof(*p));
  if (!p)
    return NM_STATUS_ALLOC_FAILED;

  p->conv = conv;
  p->algo = cfg->algo;
  p->layout = (cfg->layout == NM_LAYOUT_NHWC || cfg->layout == NM_LAYOUT_NCHW)
                  ? cfg->layout
                  : m->runtime.tensor_layout;

  if (p->layout == NM_LAYOUT_NHWC) {
    log_info("Conv2D NHWC requested; falling back to NCHW kernels for now.");
    p->layout = NM_LAYOUT_NCHW;
  }

  if (p->algo != NM_CONV_ALGO_AUTO && p->algo != NM_CONV_ALGO_DIRECT &&
      p->algo != NM_CONV_ALGO_GEMM && p->algo != NM_CONV_ALGO_REFERENCE) {
    p->algo = NM_CONV_ALGO_AUTO;
  }

  p->weight = he_init_4d(cfg->filters, cin, kh, kw);
  p->bias = st_create(1, (size_t[]){cfg->filters});
  if (!p->weight || !p->bias) {
    st_destroy(p->weight);
    st_destroy(p->bias);
    free(p);
    return NM_STATUS_ALLOC_FAILED;
  }

  NmLayerEntry *e = &m->layers[m->num_layers++];
  e->type = NM_LAYER_CONV2D;
  e->input_shape = cur;
  e->output_shape = (NmShape){.ndim = 3, .dims = {cfg->filters, out_h, out_w}};
  e->activation = cfg->activation;
  e->ops = layer_ops_for_type(e->type);
  e->params = p;
  return NM_STATUS_OK;
}

NmStatus nm_seq_add_maxpool2d(NmSeq *m, const NmPool2dConfig *cfg) {
  if (!m || !cfg || cfg->kernel_size == 0)
    return NM_STATUS_INVALID_ARGUMENT;
  if (m->compiled)
    return NM_STATUS_INVALID_ARGUMENT;

  NmShape cur = model_tail_shape(m);
  if (cur.ndim != 3) {
    log_error("MaxPool2D requires spatial input (ndim==3).");
    return NM_STATUS_INVALID_ARGUMENT;
  }

  NmStatus status = ensure_capacity(m);
  if (status != NM_STATUS_OK)
    return status;

  size_t kh = cfg->kernel_size;
  size_t kw = cfg->kernel_size;
  size_t sh = cfg->stride ? cfg->stride : kh;
  size_t sw = cfg->stride ? cfg->stride : kw;
  size_t ph = cfg->padding;
  size_t pw = cfg->padding;

  size_t in_h = cur.dims[1];
  size_t in_w = cur.dims[2];
  size_t out_h = 0;
  size_t out_w = 0;
  if (!st_pool2d_output_hw(in_h, in_w, kh, kw, sh, sw, ph, pw, &out_h,
                           &out_w)) {
    return NM_STATUS_INVALID_ARGUMENT;
  }

  NmPool2dInternalParams *p = calloc(1, sizeof(*p));
  if (!p)
    return NM_STATUS_ALLOC_FAILED;
  *p = (NmPool2dInternalParams){kh, kw, sh, sw, ph, pw};

  NmLayerEntry *e = &m->layers[m->num_layers++];
  e->type = NM_LAYER_MAXPOOL2D;
  e->input_shape = cur;
  e->output_shape = (NmShape){.ndim = 3, .dims = {cur.dims[0], out_h, out_w}};
  e->activation = NM_ACT_NONE;
  e->ops = layer_ops_for_type(e->type);
  e->params = p;
  return NM_STATUS_OK;
}

NmStatus nm_seq_add_avgpool2d(NmSeq *m, const NmPool2dConfig *cfg) {
  (void)m;
  (void)cfg;
  log_error("AvgPool2D not yet implemented — use MaxPool2D for now.");
  return NM_STATUS_UNSUPPORTED_TYPE;
}

NmStatus nm_seq_add_batchnorm2d(NmSeq *m, const NmBatchNorm2dConfig *cfg) {
  (void)m;
  (void)cfg;
  log_error("BatchNorm2D not yet implemented.");
  return NM_STATUS_UNSUPPORTED_TYPE;
}

NmStatus nm_seq_add_flatten(NmSeq *m) {
  if (!m)
    return NM_STATUS_INVALID_ARGUMENT;
  if (m->compiled)
    return NM_STATUS_INVALID_ARGUMENT;

  NmShape cur = model_tail_shape(m);
  if (cur.ndim != 3) {
    log_error("Flatten requires spatial input (ndim==3).");
    return NM_STATUS_INVALID_ARGUMENT;
  }

  NmStatus status = ensure_capacity(m);
  if (status != NM_STATUS_OK)
    return status;

  size_t flat = cur.dims[0] * cur.dims[1] * cur.dims[2];

  NmLayerEntry *e = &m->layers[m->num_layers++];
  e->type = NM_LAYER_FLATTEN;
  e->input_shape = cur;
  e->output_shape = (NmShape){.ndim = 1, .dims = {flat}};
  e->activation = NM_ACT_NONE;
  e->ops = layer_ops_for_type(e->type);
  e->params = NULL;
  return NM_STATUS_OK;
}

NmStatus nm_seq_finalize(NmSeq *m) {
  if (!m || m->num_layers == 0)
    return NM_STATUS_INVALID_ARGUMENT;
  if (m->compiled)
    return NM_STATUS_INVALID_ARGUMENT;

  NmShape tail = m->layers[m->num_layers - 1].output_shape;
  if (tail.ndim != 1) {
    log_error("Model output must be flat (ndim==1). Add Dense or Flatten.");
    return NM_STATUS_INVALID_ARGUMENT;
  }

  m->compiled = true;
  NmStatus lowering_status = nm_seq_lower_to_ir(m);
  if (lowering_status != NM_STATUS_OK) {
    m->compiled = false;
    nm_ir_graph_reset(&m->compiled_graph);
    return lowering_status;
  }
  return NM_STATUS_OK;
}

NmStatus nm_seq_prepare_engine(NmSeq *m, const NmCompileConfig *cfg) {
  if (!m || !m->compiled)
    return NM_STATUS_INVALID_ARGUMENT;

  m->compile_config = cfg ? *cfg : nm_compile_defaults();
  if (!inference_plan_reinit(&m->runtime.inference_plan))
    return NM_STATUS_ALLOC_FAILED;
  exec_plan_reset(&m->runtime.exec_plan);
  NmStatus plan_status = exec_plan_build(&m->runtime.exec_plan, &m->compiled_graph);
  if (plan_status != NM_STATUS_OK) {
    exec_plan_reset(&m->runtime.exec_plan);
    return plan_status;
  }
  bool shortcut =
      compile_uses_softmax_ce_shortcut(m, m->compile_config.loss_type);
  log_info("Model compiled — %zu layer(s), loss=%s%s", m->num_layers,
           m->compile_config.loss_type == NM_LOSS_CROSS_ENTROPY
               ? "cross-entropy"
               : "MSE",
           shortcut ? " (softmax shortcut)" : "");
  return NM_STATUS_OK;
}

NmStatus nm_seq_compile(NmSeq *m, const NmCompileConfig *cfg) {
  NmStatus status = nm_seq_finalize(m);
  return status == NM_STATUS_OK ? nm_seq_prepare_engine(m, cfg) : status;
}

void nm_seq_summary(const NmSeq *m) {
  if (!m)
    return;

  printf("\nModel Summary\n");
  printf("================================================================\n");
  printf("%-22s %-20s %s\n", "Layer (type)", "Output Shape", "Param #");
  printf("================================================================\n");

  size_t total = 0;
  for (size_t i = 0; i < m->num_layers; ++i) {
    const NmLayerEntry *e = &m->layers[i];
    char name[32];
    snprintf(name, sizeof(name), "%s-%zu", layer_type_name(e->type), i + 1);

    char shape[40];
    if (e->output_shape.ndim == 1) {
      snprintf(shape, sizeof(shape), "[%zu]", e->output_shape.dims[0]);
    } else {
      snprintf(shape, sizeof(shape), "[%zu, %zu, %zu]", e->output_shape.dims[0],
               e->output_shape.dims[1], e->output_shape.dims[2]);
    }

    size_t params = layer_param_count(e);
    total += params;
    printf("%-22s %-20s %zu\n", name, shape, params);
  }

  printf("================================================================\n");
  printf("Total params: %zu\n\n", total);
}
