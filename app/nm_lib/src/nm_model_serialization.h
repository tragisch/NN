/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/*
 * Internal serialization declarations for the nm model implementation.
 */

#ifndef NM_MODEL_SERIALIZATION_H
#define NM_MODEL_SERIALIZATION_H

#include "app/nm_lib/src/nm_model_internal.h"

#define NM_SEQ_BIN_MAGIC "NMS3"
#define NM_SEQ_BIN_VERSION 4U

#define NM_SEQ_BIN_VERSION_V3 3U
#define NM_SEQ_BIN_MAGIC_LEGACY "NMS2"
#define NM_SEQ_BIN_VERSION_LEGACY 2U

#define NM_SEQ_ALIGNMENT 32U
#define NM_SEQ_FLAG_INFERENCE_ONLY 0x00000001U

#define NM_SEQ_CHUNK_CONF "CONF"
#define NM_SEQ_CHUNK_META "META"
#define NM_SEQ_CHUNK_LAYS "LAYS"
#define NM_SEQ_CHUNK_TENS "TENS"
#define NM_SEQ_CHUNK_WGHT "WGHT"
#define NM_SEQ_CHUNK_WGHZ "WGHZ"
#define NM_SEQ_CHUNK_PLAN "PLAN"
#define NM_SEQ_CHUNK_TRNG "TRNG"
#define NM_SEQ_CHUNK_OPTS "OPTS"
#define NM_SEQ_CHUNK_HASH "HASH"

#define NM_SER_LAYER_NAME_MAX 64
#define NM_SER_TENSOR_NAME_MAX 96

typedef enum {
  NM_SER_TENSOR_WEIGHT = 0,
  NM_SER_TENSOR_BIAS = 1,
} NmSerTensorRole;

typedef struct {
  char name[NM_SER_LAYER_NAME_MAX];
  NmLayerType type;
  NmActivation activation;
  NmShape input_shape;
  NmShape output_shape;
  struct {
    size_t units;
    size_t filters;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    NmConvAlgo algo;
    NmTensorLayout layout;
    size_t kernel_h, kernel_w;
    size_t stride_h, stride_w;
    size_t pad_h, pad_w;
  } cfg;
} NmSerLayerDesc;

typedef struct {
  char name[NM_SER_TENSOR_NAME_MAX];
  char layer_name[NM_SER_LAYER_NAME_MAX];
  uint32_t layer_index;
  uint32_t role;
  uint32_t ndim;
  uint64_t dims[4];
  uint64_t data_bytes;
  uint8_t dtype;
} NmSerTensorDesc;

typedef struct {
  uint32_t alignment;
  uint32_t flags;
  uint32_t format_version;
  uint64_t optimizer_step;
  uint64_t plan_batch_rows;
  uint64_t plan_output_cols;
  uint64_t plan_peak_tensors;
  uint64_t plan_peak_bytes;
  NmShape input_shape;
  NmCompileConfig compile_config;
  NmOptimizerConfig optimizer_config;
  NmTensorLayout layout;
  NmAutogradMode autograd_mode;
  NmBatchParallelConfig batch_parallel;
  char backend_name[64];
  char architecture[32];
  NmSerLayerDesc *layers;
  size_t layer_count;
  NmSerTensorDesc *tensors;
  size_t tensor_count;
} NmSerModelDesc;

#endif /* NM_MODEL_SERIALIZATION_H */
