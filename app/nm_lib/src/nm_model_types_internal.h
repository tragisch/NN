/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/*
 * Internal type declarations for the nm model implementation.
 */

#ifndef NM_MODEL_TYPES_INTERNAL_H
#define NM_MODEL_TYPES_INTERNAL_H

#include "nm.h"
#include "nm_graph_ir_internal.h"
#include "st.h"
#include "st_conv.h"
#include "st_pool.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NmSeq NmSeq;
typedef struct NmLayerEntry NmLayerEntry;
typedef struct NmBackendOps NmBackendOps;

typedef struct {
  float *m;
  float *v;
  size_t size;
} NmOptSlot;

typedef struct {
  FloatMatrix *weights; /* [input_dim, units] */
  FloatMatrix *bias;    /* [1, units]         */
  NmOptSlot w_slot;
  NmOptSlot b_slot;
} NmDenseParams;

typedef struct {
  FloatTensor *weight; /* [Cout, Cin, Kh, Kw] */
  FloatTensor *bias;   /* [Cout]              */
  StConv2dParams conv;
  NmConvAlgo algo;
  NmTensorLayout layout;
  NmOptSlot w_slot;
  NmOptSlot b_slot;
} NmConv2dInternalParams;

typedef struct {
  size_t kernel_h, kernel_w;
  size_t stride_h, stride_w;
  size_t pad_h, pad_w;
} NmPool2dInternalParams;

typedef struct {
  FloatTensor *tensor;
  bool in_use;
} NmTensorPoolEntry;

typedef struct {
  NmTensorPoolEntry *entries;
  size_t count;
  size_t capacity;
} NmTensorPool;

typedef struct {
  NmLayerType layer_type;
  NmShape output_shape;
  size_t tensor_numel;
} NmExecTraceNode;

typedef struct {
  NmExecTraceNode *nodes;
  size_t count;
  size_t capacity;
} NmExecTrace;

typedef enum {
  NM_EXEC_STEP_DENSE = 0,      /* MatMul + Add (+ activation) */
  NM_EXEC_STEP_CONV2D = 1,     /* Conv2D (+ activation) */
  NM_EXEC_STEP_MAXPOOL2D = 2,
  NM_EXEC_STEP_FLATTEN = 3,    /* Reshape to [batch, features] */
  NM_EXEC_STEP_ACTIVATION = 4, /* stand-alone ReLU / Sigmoid / Tanh / Softmax */
} NmExecPlanStepKind;

typedef struct {
  NmExecPlanStepKind kind;
  uint32_t node_id;      /* node that carries the attributes */
  uint32_t input_value;
  uint32_t output_value; /* last value produced by the fused nodes */
  uint32_t weight_value; /* NM_IR_INVALID_ID when unused */
  uint32_t bias_value;
  NmActivation activation;
} NmExecPlanStep;

typedef struct {
  NmExecPlanStep *steps;
  size_t count;
  size_t capacity;
  uint32_t input_value_id;
  uint32_t model_output_id;
  NmShape input_shape; /* without the batch dimension */
  size_t input_cols;
  size_t output_cols;
  bool dense_only;
} NmExecPlan;

typedef struct {
  NmOptimizerConfig optimizer_config;
  uint64_t optimizer_step;
  NmAutogradMode autograd_mode;
  NmBatchParallelConfig batch_parallel;
  NmExecTrace last_trace;
  bool inference_only;
} NmTrainingState;

typedef struct {
  NmTensorPool pool;
  size_t prepared_batch_rows;
  size_t output_cols;
  size_t peak_tensor_count;
  size_t peak_tensor_bytes;
  bool prepared;
} NmInferencePlan;

typedef struct {
  const NmBackendOps *backend_ops;
  const NmBackendCapabilities *backend_caps;
  const char *backend_name;
  NmTensorLayout tensor_layout;
  NmExecPlan exec_plan;
  NmInferencePlan inference_plan;
} NmRuntimeState;

typedef struct {
  uint32_t format_version;
  uint32_t alignment;
  bool inference_only;
  bool loaded_as_bf16;
} NmSerializationState;

typedef struct {
  const FloatTensor *input;
  FloatTensor **out_pool_idx;
  NmTensorPool *pool;
  const NmBackendOps *backend;
} NmLayerFwdCtx;

typedef struct {
  const FloatTensor *input;
  const FloatTensor *activation;
  FloatTensor *grad;
  FloatTensor *pool_indices;
  bool skip_act;
  const NmOptimizerConfig *opt_cfg;
  uint64_t opt_step;
  NmTensorPool *pool;
  const NmBackendOps *backend;
} NmLayerBwdCtx;

typedef struct {
  FloatTensor *(*forward)(const NmLayerEntry *layer, const NmLayerFwdCtx *ctx);
  FloatTensor *(*backward)(NmLayerEntry *layer, const NmLayerBwdCtx *ctx);
} NmLayerOps;

struct NmBackendOps {
  bool (*gemm)(FloatMatrix *c, float alpha, const FloatMatrix *a,
               SmTranspose ta, const FloatMatrix *b, SmTranspose tb,
               float beta);
  bool (*gemm_bias_relu)(FloatMatrix *out, const FloatMatrix *in,
                         SmTranspose ta, const FloatMatrix *w, SmTranspose tw,
                         const FloatMatrix *bias);
  bool (*gemm_bias_act)(FloatMatrix *out, const FloatMatrix *in,
                        SmTranspose ta, const FloatMatrix *w, SmTranspose tw,
                        const FloatMatrix *bias, SmActivation act);
  void (*add_rowwise)(FloatMatrix *mat, const FloatMatrix *row);
  void (*softmax)(FloatMatrix *mat);
  bool (*conv2d_fwd)(const FloatTensor *input, const FloatTensor *weight,
                     const FloatTensor *bias, const StConv2dParams *params,
                     FloatTensor *output);
  bool (*conv2d_bwd_data)(const FloatTensor *grad_output,
                          const FloatTensor *weight,
                          const StConv2dParams *params,
                          FloatTensor *grad_input);
  void (*conv2d_bwd_weight)(const FloatTensor *input,
                            const FloatTensor *grad_output,
                            const StConv2dParams *params,
                            FloatTensor *grad_weight);
  void (*conv2d_bwd_bias)(const FloatTensor *grad_output,
                          FloatTensor *grad_bias);
  bool (*maxpool2d_fwd)(const FloatTensor *input, size_t kernel_h,
                        size_t kernel_w, size_t stride_h, size_t stride_w,
                        size_t pad_h, size_t pad_w, FloatTensor *output,
                        FloatTensor *indices);
  bool (*maxpool2d_bwd)(const FloatTensor *grad_output,
                        const FloatTensor *indices, size_t in_h, size_t in_w,
                        FloatTensor *grad_input);
};

struct NmLayerEntry {
  NmLayerType type;
  NmShape input_shape;
  NmShape output_shape;
  NmActivation activation;
  const NmLayerOps *ops;
  void *params; /* type-specific; NULL for Flatten */
};

struct NmSeq {
  NmLayerEntry *layers;
  size_t num_layers;
  size_t capacity;
  NmShape input_shape;
  NmCompileConfig compile_config;
  NmRuntimeState runtime;
  NmTrainingState training;
  NmSerializationState storage;
  NmIrGraph compiled_graph;
  bool compiled;
};

/* What the training loops need from a model: layers with their parameters,
 * the loss, the optimizer state and the backend. Built from an NmSeq or from
 * an engine that runs a graph. */
typedef struct {
  NmLayerEntry *layers;
  size_t num_layers;
  NmShape input_shape;
  NmLoss loss_type;
  NmTrainingState *training;
  const NmBackendOps *backend;
} NmTrainable;

/* Layer entries for the steps of an execution plan. Parameters point at the
 * graph's constants, so training updates them in place; optimizer state lives
 * here. For an NmSeq the entries are the model's own layers (`owned` false). */
typedef struct {
  NmLayerEntry *layers;
  size_t count;
  void *params; /* per-layer parameter storage, NULL when not owned */
  bool owned;
} NmLayerRuntime;


#ifdef __cplusplus
}
#endif

#endif /* NM_MODEL_TYPES_INTERNAL_H */
