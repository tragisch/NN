/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/*
 * Shared public types for the nm frontend, C engine and exporters.
 */
#ifndef NM_TYPES_H
#define NM_TYPES_H

#define NM_SHARED_TYPES_DEFINED 1

#include "sm.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Executable model graph (GraphIR-shaped), see nm_import_graphir.h. */
typedef struct NmIrGraph NmGraphModel;

typedef enum NmStatus {
  NM_STATUS_OK = 0,
  NM_STATUS_INVALID_ARGUMENT = 1,
  NM_STATUS_IO_ERROR = 2,
  NM_STATUS_ALLOC_FAILED = 3,
  NM_STATUS_FORMAT_ERROR = 4,
  NM_STATUS_UNSUPPORTED_TYPE = 5,
  NM_STATUS_INTERNAL_ERROR = 6,
} NmStatus;

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

#include "nm_input.h"

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

#ifdef __cplusplus
}
#endif

#endif /* NM_TYPES_H */
