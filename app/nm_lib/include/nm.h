/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef NM_H
#define NM_H

#include "sm.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "nm_input.h"

#ifndef NM_SHARED_TYPES_DEFINED
typedef enum NmStatus {
  NM_STATUS_OK = 0,
  NM_STATUS_INVALID_ARGUMENT = 1,
  NM_STATUS_IO_ERROR = 2,
  NM_STATUS_ALLOC_FAILED = 3,
  NM_STATUS_FORMAT_ERROR = 4,
  NM_STATUS_UNSUPPORTED_TYPE = 5,
  NM_STATUS_INTERNAL_ERROR = 6,
} NmStatus;

/**************************************/
/*            Core datatypes          */
/**************************************/

typedef enum {
  NM_ACT_NONE = 0,
  NM_ACT_RELU = 1,
  NM_ACT_SIGMOID = 2,
  NM_ACT_TANH = 3,
  NM_ACT_SOFTMAX = 4,
} NmActivation;

typedef enum {
  NM_LOSS_NONE = 0,
  NM_LOSS_MSE = 1,
  NM_LOSS_CROSS_ENTROPY = 2,
} NmLoss;

typedef struct {
  NmLoss loss_type;
} NmCompileConfig;

typedef struct {
  size_t batch_size;
  float learning_rate;
  size_t epochs;
  bool verbose;
} NmTrainConfig;

typedef struct {
  size_t ndim;
  size_t dims[3];
} NmShape;

typedef enum {
  NM_LAYER_DENSE,
  NM_LAYER_CONV2D,
  NM_LAYER_MAXPOOL2D,
  NM_LAYER_AVGPOOL2D,
  NM_LAYER_BATCHNORM2D,
  NM_LAYER_FLATTEN,
} NmLayerType;

typedef enum {
  NM_LAYOUT_NCHW = 0,
  NM_LAYOUT_NHWC = 1,
} NmTensorLayout;

typedef enum {
  NM_CONV_ALGO_AUTO = 0,
  NM_CONV_ALGO_DIRECT = 1,
  NM_CONV_ALGO_GEMM = 2,
  NM_CONV_ALGO_REFERENCE = 3,
} NmConvAlgo;

typedef struct {
  size_t units;
  NmActivation activation;
} NmDenseConfig;

typedef struct {
  size_t filters;
  size_t kernel_size;
  size_t stride;
  size_t padding;
  NmConvAlgo algo;
  NmTensorLayout layout;
  NmActivation activation;
} NmConv2dConfig;

typedef struct {
  size_t kernel_size;
  size_t stride;
  size_t padding;
} NmPool2dConfig;

typedef struct {
  float epsilon;
} NmBatchNorm2dConfig;

typedef enum {
  NM_OPT_SGD = 0,
  NM_OPT_MOMENTUM = 1,
  NM_OPT_RMSPROP = 2,
  NM_OPT_ADAM = 3,
} NmOptimizerType;

typedef struct {
  NmOptimizerType type;
  float learning_rate;
  float momentum;
  float beta1;
  float beta2;
  float epsilon;
} NmOptimizerConfig;

typedef enum {
  NM_BACKEND_AUTO = 0,
  NM_BACKEND_DEFAULT = 1,
} NmBackendType;

typedef enum {
  NM_AUTOGRAD_DISABLED = 0,
  NM_AUTOGRAD_TAPE = 1,
} NmAutogradMode;

typedef struct {
  bool enabled;
  bool deterministic;
  size_t num_workers;
} NmBatchParallelConfig;

typedef struct {
  const char *name;
  bool deterministic;
  bool supports_training;
  bool supports_autograd_tape;
  bool supports_tensor_pool_reuse;
  bool supports_prepared_inference;
  bool supports_fused_dense_relu;
  bool supports_conv_gemm;
  bool prefers_prepared_inference;
} NmBackendCapabilities;

#endif /* NM_SHARED_TYPES_DEFINED */

typedef struct NmSeq NmModel;

/**************************************/
/*          Config constructors       */
/*  All return stack values (no heap) */
/**************************************/

NmCompileConfig nm_compile_defaults(void);
NmTrainConfig nm_train_defaults(void);
NmOptimizerConfig nm_optimizer_defaults(void);

/**************************************/
/*           Model lifecycle          */
/**************************************/

/* Returns newly allocated model. Caller owns the result and must call
 * nm_free() when done. Returns NULL on invalid input or alloc failure. */
NmModel *nm_create(NmShape input_shape);

/* Frees model and all owned resources (layers, weights, optimizer state).
 * Accepts NULL safely. */
void nm_free(NmModel *model);

/**************************************/
/*            Model builder           */
/* All functions borrow the model ptr */
/* and take ownership of no arguments */
/**************************************/

NmStatus nm_add_dense(NmModel *m, const NmDenseConfig *cfg);
NmStatus nm_add_conv2d(NmModel *m, const NmConv2dConfig *cfg);
NmStatus nm_add_maxpool2d(NmModel *m, const NmPool2dConfig *cfg);
NmStatus nm_add_avgpool2d(NmModel *m, const NmPool2dConfig *cfg);
NmStatus nm_add_batchnorm2d(NmModel *m, const NmBatchNorm2dConfig *cfg);
NmStatus nm_add_flatten(NmModel *m);

/**************************************/
/*      Compile / runtime control     */
/**************************************/

NmStatus nm_compile(NmModel *m, const NmCompileConfig *cfg);
NmStatus nm_set_optimizer(NmModel *m, const NmOptimizerConfig *cfg);
NmStatus nm_set_backend(NmModel *m, NmBackendType backend);

/* Returns borrowed pointer valid for the lifetime of the model. */
const char *nm_model_backend(const NmModel *m);

/* Returns borrowed static capability descriptor for the active process-wide
 * backend. Caller must not free or modify the returned pointer. */
const NmBackendCapabilities *nm_backend_capabilities(void);

/* Returns borrowed capability descriptor for the model runtime backend.
 * Pointer remains valid for the lifetime of the model. */
const NmBackendCapabilities *nm_model_backend_capabilities(const NmModel *m);

NmStatus nm_set_tensor_layout(NmModel *m, NmTensorLayout layout);
NmTensorLayout nm_tensor_layout(const NmModel *m);
NmStatus nm_set_autograd_mode(NmModel *m, NmAutogradMode mode);
NmAutogradMode nm_autograd_mode(const NmModel *m);
NmStatus nm_set_batch_parallel(NmModel *m, const NmBatchParallelConfig *cfg);
NmBatchParallelConfig nm_batch_parallel(const NmModel *m);
size_t nm_last_graph_nodes(const NmModel *m);

/**************************************/
/*       Training / inference         */
/**************************************/

/* X and Y are borrowed (read-only). Model weights are updated in-place. */
NmStatus nm_fit_epoch(NmModel *m, const FloatMatrix *X, const FloatMatrix *Y,
                      const NmTrainConfig *cfg);
NmStatus nm_fit(NmModel *m, const FloatMatrix *X, const FloatMatrix *Y,
                const NmTrainConfig *cfg);

/* Returns newly allocated matrix in *out_prediction. Caller owns it and
 * must call sm_destroy(). Input is borrowed. */
NmStatus nm_predict(const NmModel *m, const FloatMatrix *input,
                    FloatMatrix **out_prediction);

/* Writes into caller-supplied out_prediction (borrowed, must be pre-allocated
 * with correct dimensions). */
NmStatus nm_predict_into(const NmModel *m, const FloatMatrix *input,
                         FloatMatrix *out_prediction);

NmStatus nm_evaluate(const NmModel *m, const FloatMatrix *X,
                     const FloatMatrix *Y, float *out_accuracy);
void nm_summary(const NmModel *m);

/**************************************/
/*            Serialization           */
/**************************************/

/* Writes model to file at path. Model is borrowed. */
NmStatus nm_save(const NmModel *m, const char *path);

/* Writes an inference-oriented snapshot that omits training-only runtime state
 * such as optimizer progress. Model is borrowed. The exported file can be
 * loaded with nm_load() but should be treated as inference-only afterwards. */
NmStatus nm_save_inference(const NmModel *m, const char *path);

/* Options for nm_save_inference_ex(). */
typedef struct {
  bool use_bf16; /* Export weights as bfloat16 (halves file size). */
} NmSaveInferenceOptions;

/* Extended inference export with options. When use_bf16 is true, weights are
 * converted to bfloat16 on the fly; the in-RAM model stays f32.
 * Loaded bf16 models are auto-promoted to f32 and marked inference-only. */
NmStatus nm_save_inference_ex(const NmModel *m, const char *path,
                              const NmSaveInferenceOptions *opts);

/* Returns newly allocated model in *out_model. Caller owns it. */
NmStatus nm_load(const char *path, NmModel **out_model);

/* Returns true when the model was loaded from or exported as an inference-only
 * snapshot and therefore should not be used for training. */
bool nm_model_is_inference_only(const NmModel *m);

/* Returns true when the model was loaded from a file that stored weights in
 * bfloat16 format (they are auto-promoted to f32 on load). */
bool nm_model_was_bf16(const NmModel *m);

/**************************************/
/*             Math helpers           */
/**************************************/

NmStatus nm_accuracy(const FloatMatrix *predicted, const FloatMatrix *target,
                     float *out_accuracy);

/* Writes result into caller-supplied out_result (borrowed). */
NmStatus nm_linear_into(const FloatMatrix *input, const FloatMatrix *weights,
                        const FloatMatrix *bias, FloatMatrix *out_result);

/* Returns newly allocated matrix in *out_result. Caller owns it. */
NmStatus nm_linear(const FloatMatrix *input, const FloatMatrix *weights,
                   const FloatMatrix *bias, FloatMatrix **out_result);

/* In-place activation functions. Input is modified (borrowed, not freed). */
void nm_relu(FloatMatrix *mat);
void nm_sigmoid(FloatMatrix *mat);
void nm_tanh(FloatMatrix *mat);
void nm_softmax(FloatMatrix *mat);

/* In-place gradient functions. grad_output is modified; activation is
 * borrowed read-only. */
void nm_relu_grad(const FloatMatrix *activation, FloatMatrix *grad_output);
void nm_sigmoid_grad(const FloatMatrix *activation, FloatMatrix *grad_output);
void nm_tanh_grad(const FloatMatrix *activation, FloatMatrix *grad_output);
void nm_softmax_ce_grad(const FloatMatrix *predicted, const FloatMatrix *target,
                        FloatMatrix *grad_output);

/* Loss functions. All inputs are borrowed. Returns negative value on error. */
float nm_loss_mse(const FloatMatrix *predicted, const FloatMatrix *target);
float nm_loss_cross_entropy(const FloatMatrix *predicted,
                            const FloatMatrix *target);

/* In-place row-wise addition. mat is modified; row is borrowed. */
void nm_add_rowwise(FloatMatrix *mat, const FloatMatrix *row);

/* Returns newly allocated matrix. Caller owns it. */
NmStatus nm_sum_rows(const FloatMatrix *mat, FloatMatrix **out_sum_rows);

/* Returns newly allocated matrix. Caller owns it. */
NmStatus nm_argmax_rows(const FloatMatrix *mat, FloatMatrix **out_argmax);

float nm_tanhf(float x);
float nm_softmax_denom(const float *vec, size_t len);

/**************************************/
/*           Utility helpers          */
/**************************************/

/* Returns borrowed static string identifying the active BLAS backend. */
const char *nm_backend(void);

void nm_set_seed(uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif /* NM_H */
