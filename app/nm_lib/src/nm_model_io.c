/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/*
 * Serialization and persistence for nm models.
 */

#include "nm_model_serialization.h"
#include "st_dtype.h"
#include <log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xxhash.h>
#include <zstd.h>

static bool io_write_exact(FILE *f, const void *ptr, size_t bytes) {
  return f && ptr && fwrite(ptr, 1, bytes, f) == bytes;
}

static bool io_read_exact(FILE *f, void *ptr, size_t bytes) {
  return f && ptr && fread(ptr, 1, bytes, f) == bytes;
}

static bool io_write_u8(FILE *f, uint8_t v) {
  return io_write_exact(f, &v, sizeof(v));
}
static bool io_read_u8(FILE *f, uint8_t *v) {
  return io_read_exact(f, v, sizeof(*v));
}
static bool io_write_u32(FILE *f, uint32_t v) {
  return io_write_exact(f, &v, sizeof(v));
}
static bool io_read_u32(FILE *f, uint32_t *v) {
  return io_read_exact(f, v, sizeof(*v));
}
static bool io_write_u64(FILE *f, uint64_t v) {
  return io_write_exact(f, &v, sizeof(v));
}
static bool io_read_u64(FILE *f, uint64_t *v) {
  return io_read_exact(f, v, sizeof(*v));
}
static bool io_write_f32(FILE *f, float v) {
  return io_write_exact(f, &v, sizeof(v));
}
static bool io_read_f32(FILE *f, float *v) {
  return io_read_exact(f, v, sizeof(*v));
}

static bool io_write_shape(FILE *f, NmShape s) {
  return io_write_u64(f, (uint64_t)s.ndim) &&
         io_write_u64(f, (uint64_t)s.dims[0]) &&
         io_write_u64(f, (uint64_t)s.dims[1]) &&
         io_write_u64(f, (uint64_t)s.dims[2]);
}

static bool io_read_shape(FILE *f, NmShape *out) {
  if (!out)
    return false;
  uint64_t ndim = 0;
  uint64_t d0 = 0;
  uint64_t d1 = 0;
  uint64_t d2 = 0;
  if (!io_read_u64(f, &ndim) || !io_read_u64(f, &d0) || !io_read_u64(f, &d1) ||
      !io_read_u64(f, &d2)) {
    return false;
  }
  if (ndim != 1 && ndim != 3)
    return false;
  out->ndim = (size_t)ndim;
  out->dims[0] = (size_t)d0;
  out->dims[1] = (size_t)d1;
  out->dims[2] = (size_t)d2;
  return true;
}

static bool io_skip_bytes(FILE *f, uint64_t bytes) {
  uint8_t scratch[4096];
  while (bytes > 0) {
    size_t step =
        (bytes > (uint64_t)sizeof(scratch)) ? sizeof(scratch) : (size_t)bytes;
    if (fread(scratch, 1, step, f) != step)
      return false;
    bytes -= (uint64_t)step;
  }
  return true;
}

static bool io_write_padding(FILE *f, uint64_t bytes) {
  static const uint8_t zeros[32] = {0};
  while (bytes > 0) {
    size_t step =
        bytes > (uint64_t)sizeof(zeros) ? sizeof(zeros) : (size_t)bytes;
    if (fwrite(zeros, 1, step, f) != step)
      return false;
    bytes -= (uint64_t)step;
  }
  return true;
}

static uint64_t io_string_size(const char *s) {
  size_t len = s ? strlen(s) : 0;
  return (uint64_t)sizeof(uint16_t) + (uint64_t)len;
}

static bool io_write_string(FILE *f, const char *s) {
  uint16_t len = (uint16_t)(s ? strlen(s) : 0);
  return io_write_exact(f, &len, sizeof(len)) &&
         (len == 0 || io_write_exact(f, s, len));
}

static bool io_read_string(FILE *f, char *dst, size_t dst_size) {
  uint16_t len = 0;
  if (!f || !dst || dst_size == 0 || !io_read_exact(f, &len, sizeof(len)))
    return false;
  if ((size_t)len >= dst_size)
    return false;
  if (len > 0 && !io_read_exact(f, dst, len))
    return false;
  dst[len] = '\0';
  return true;
}

static const char *layer_type_slug(NmLayerType type) {
  switch (type) {
  case NM_LAYER_DENSE:
    return "dense";
  case NM_LAYER_CONV2D:
    return "conv2d";
  case NM_LAYER_MAXPOOL2D:
    return "maxpool2d";
  case NM_LAYER_AVGPOOL2D:
    return "avgpool2d";
  case NM_LAYER_BATCHNORM2D:
    return "batchnorm2d";
  case NM_LAYER_FLATTEN:
    return "flatten";
  default:
    return "layer";
  }
}

static void layer_serial_name(const NmLayerEntry *e, size_t index, char *out,
                              size_t out_size) {
  if (!out || out_size == 0)
    return;
  snprintf(out, out_size, "blk.%zu.%s", index, layer_type_slug(e->type));
}

static void tensor_serial_name(const char *layer_name, const char *suffix,
                               char *out, size_t out_size) {
  if (!out || out_size == 0)
    return;
  snprintf(out, out_size, "%s.%s", layer_name ? layer_name : "blk.0.layer",
           suffix ? suffix : "tensor");
}

static size_t nm_seq_tensor_desc_count(const NmSeq *m) {
  size_t count = 0;
  if (!m)
    return 0;
  for (size_t i = 0; i < m->num_layers; ++i) {
    switch (m->layers[i].type) {
    case NM_LAYER_DENSE:
    case NM_LAYER_CONV2D:
      count += 2;
      break;
    default:
      break;
    }
  }
  return count;
}

static NmStatus collect_tensor_descs_ex(const NmSeq *m, StDtype export_dtype,
                                        NmSerTensorDesc **out_descs,
                                        size_t *out_count);

static void ser_model_desc_init(NmSerModelDesc *desc) {
  if (!desc)
    return;
  memset(desc, 0, sizeof(*desc));
  desc->alignment = NM_SEQ_ALIGNMENT;
  desc->format_version = NM_SEQ_BIN_VERSION;
  desc->compile_config = nm_compile_defaults();
  desc->optimizer_config = optimizer_defaults_internal();
  desc->layout = NM_LAYOUT_NCHW;
  desc->autograd_mode = NM_AUTOGRAD_DISABLED;
  desc->batch_parallel.enabled = false;
  desc->batch_parallel.deterministic = true;
  desc->batch_parallel.num_workers = 0;
  snprintf(desc->architecture, sizeof(desc->architecture), "%s", "nm.seq");
}

static void ser_model_desc_destroy(NmSerModelDesc *desc) {
  if (!desc)
    return;
  free(desc->layers);
  free(desc->tensors);
  ser_model_desc_init(desc);
}

static NmStatus collect_layer_descs(const NmSeq *m, NmSerLayerDesc **out_descs,
                                    size_t *out_count) {
  if (!m || !out_descs || !out_count)
    return NM_STATUS_INVALID_ARGUMENT;
  *out_descs = NULL;
  *out_count = 0;

  if (m->num_layers == 0)
    return NM_STATUS_OK;

  NmSerLayerDesc *descs = calloc(m->num_layers, sizeof(*descs));
  if (!descs)
    return NM_STATUS_ALLOC_FAILED;

  for (size_t i = 0; i < m->num_layers; ++i) {
    const NmLayerEntry *e = &m->layers[i];
    NmSerLayerDesc *d = &descs[i];
    layer_serial_name(e, i, d->name, sizeof(d->name));
    d->type = e->type;
    d->activation = e->activation;
    d->input_shape = e->input_shape;
    d->output_shape = e->output_shape;

    switch (e->type) {
    case NM_LAYER_DENSE: {
      const NmDenseParams *p = (const NmDenseParams *)e->params;
      d->cfg.units = p->weights->cols;
      break;
    }
    case NM_LAYER_CONV2D: {
      const NmConv2dInternalParams *p =
          (const NmConv2dInternalParams *)e->params;
      d->cfg.filters = p->weight->shape[0];
      d->cfg.kernel_size = p->weight->shape[2];
      d->cfg.stride = p->conv.stride_h;
      d->cfg.padding = p->conv.pad_h;
      d->cfg.algo = p->algo;
      d->cfg.layout = p->layout;
      break;
    }
    case NM_LAYER_MAXPOOL2D: {
      const NmPool2dInternalParams *p =
          (const NmPool2dInternalParams *)e->params;
      d->cfg.kernel_h = p->kernel_h;
      d->cfg.kernel_w = p->kernel_w;
      d->cfg.stride_h = p->stride_h;
      d->cfg.stride_w = p->stride_w;
      d->cfg.pad_h = p->pad_h;
      d->cfg.pad_w = p->pad_w;
      break;
    }
    case NM_LAYER_FLATTEN:
      break;
    default:
      free(descs);
      return NM_STATUS_UNSUPPORTED_TYPE;
    }
  }

  *out_descs = descs;
  *out_count = m->num_layers;
  return NM_STATUS_OK;
}

static NmStatus ser_model_desc_from_model(const NmSeq *m, bool inference_only,
                                          StDtype export_dtype,
                                          NmSerModelDesc *out) {
  if (!m || !out)
    return NM_STATUS_INVALID_ARGUMENT;

  NmSerModelDesc desc;
  ser_model_desc_init(&desc);
  desc.input_shape = m->input_shape;
  desc.compile_config = m->compile_config;
  desc.optimizer_config = inference_only ? optimizer_defaults_internal()
                                         : m->training.optimizer_config;
  desc.layout = m->runtime.tensor_layout;
  desc.autograd_mode =
      inference_only ? NM_AUTOGRAD_DISABLED : m->training.autograd_mode;
  desc.batch_parallel = inference_only
                            ? (NmBatchParallelConfig){.enabled = false,
                                                      .deterministic = true,
                                                      .num_workers = 0}
                            : m->training.batch_parallel;
  desc.flags = inference_only ? NM_SEQ_FLAG_INFERENCE_ONLY : 0U;
  desc.optimizer_step = inference_only ? 0 : m->training.optimizer_step;
  desc.plan_batch_rows = m->runtime.inference_plan.prepared_batch_rows;
  desc.plan_output_cols = m->runtime.inference_plan.output_cols;
  desc.plan_peak_tensors = m->runtime.inference_plan.peak_tensor_count;
  desc.plan_peak_bytes = m->runtime.inference_plan.peak_tensor_bytes;
  snprintf(desc.backend_name, sizeof(desc.backend_name), "%s",
           m->runtime.backend_name ? m->runtime.backend_name : "unknown");

  NmStatus status = collect_layer_descs(m, &desc.layers, &desc.layer_count);
  if (status != NM_STATUS_OK) {
    ser_model_desc_destroy(&desc);
    return status;
  }

  status = collect_tensor_descs_ex(m, export_dtype, &desc.tensors,
                                   &desc.tensor_count);
  if (status != NM_STATUS_OK) {
    ser_model_desc_destroy(&desc);
    return status;
  }

  ser_model_desc_destroy(out);
  *out = desc;
  return NM_STATUS_OK;
}

static NmStatus collect_tensor_descs_ex(const NmSeq *m, StDtype export_dtype,
                                        NmSerTensorDesc **out_descs,
                                        size_t *out_count) {
  if (!m || !out_descs || !out_count)
    return NM_STATUS_INVALID_ARGUMENT;
  *out_descs = NULL;
  *out_count = 0;

  size_t count = nm_seq_tensor_desc_count(m);
  if (count == 0)
    return NM_STATUS_OK;

  NmSerTensorDesc *descs = calloc(count, sizeof(*descs));
  if (!descs)
    return NM_STATUS_ALLOC_FAILED;

  size_t w = 0;
  const size_t elem_size = st_dtype_size(export_dtype);
  for (size_t i = 0; i < m->num_layers; ++i) {
    const NmLayerEntry *e = &m->layers[i];
    char layer_name[NM_SER_LAYER_NAME_MAX];
    layer_serial_name(e, i, layer_name, sizeof(layer_name));

    if (e->type == NM_LAYER_DENSE) {
      const NmDenseParams *p = (const NmDenseParams *)e->params;
      tensor_serial_name(layer_name, "weight", descs[w].name,
                         sizeof(descs[w].name));
      snprintf(descs[w].layer_name, sizeof(descs[w].layer_name), "%s",
               layer_name);
      descs[w].layer_index = (uint32_t)i;
      descs[w].role = NM_SER_TENSOR_WEIGHT;
      descs[w].ndim = 2U;
      descs[w].dims[0] = (uint64_t)p->weights->rows;
      descs[w].dims[1] = (uint64_t)p->weights->cols;
      descs[w].data_bytes =
          (uint64_t)(p->weights->rows * p->weights->cols * elem_size);
      descs[w].dtype = (uint8_t)export_dtype;
      ++w;

      tensor_serial_name(layer_name, "bias", descs[w].name,
                         sizeof(descs[w].name));
      snprintf(descs[w].layer_name, sizeof(descs[w].layer_name), "%s",
               layer_name);
      descs[w].layer_index = (uint32_t)i;
      descs[w].role = NM_SER_TENSOR_BIAS;
      descs[w].ndim = 2U;
      descs[w].dims[0] = (uint64_t)p->bias->rows;
      descs[w].dims[1] = (uint64_t)p->bias->cols;
      descs[w].data_bytes = (uint64_t)(p->bias->cols * elem_size);
      descs[w].dtype = (uint8_t)export_dtype;
      ++w;
    } else if (e->type == NM_LAYER_CONV2D) {
      const NmConv2dInternalParams *p =
          (const NmConv2dInternalParams *)e->params;
      tensor_serial_name(layer_name, "weight", descs[w].name,
                         sizeof(descs[w].name));
      snprintf(descs[w].layer_name, sizeof(descs[w].layer_name), "%s",
               layer_name);
      descs[w].layer_index = (uint32_t)i;
      descs[w].role = NM_SER_TENSOR_WEIGHT;
      descs[w].ndim = 4U;
      for (size_t d = 0; d < 4; ++d)
        descs[w].dims[d] = (uint64_t)p->weight->shape[d];
      descs[w].data_bytes = (uint64_t)(p->weight->numel * elem_size);
      descs[w].dtype = (uint8_t)export_dtype;
      ++w;

      tensor_serial_name(layer_name, "bias", descs[w].name,
                         sizeof(descs[w].name));
      snprintf(descs[w].layer_name, sizeof(descs[w].layer_name), "%s",
               layer_name);
      descs[w].layer_index = (uint32_t)i;
      descs[w].role = NM_SER_TENSOR_BIAS;
      descs[w].ndim = 1U;
      descs[w].dims[0] = (uint64_t)p->bias->numel;
      descs[w].data_bytes = (uint64_t)(p->bias->numel * elem_size);
      descs[w].dtype = (uint8_t)export_dtype;
      ++w;
    }
  }

  *out_descs = descs;
  *out_count = w;
  return NM_STATUS_OK;
}

static uint64_t nm_seq_conf_chunk_size(void) { return 87ULL; }

static uint64_t nm_seq_layer_desc_size(const NmSerLayerDesc *d) {
  if (!d)
    return 0;
  uint64_t size = 4 + 4 + 32 + 32;
  switch (d->type) {
  case NM_LAYER_DENSE:
    size += 8;
    break;
  case NM_LAYER_CONV2D:
    size += 8 + 8 + 8 + 8 + 4 + 4;
    break;
  case NM_LAYER_MAXPOOL2D:
    size += 8 + 8 + 8 + 8 + 8 + 8;
    break;
  case NM_LAYER_FLATTEN:
    break;
  default:
    break;
  }
  return size;
}

static bool io_write_chunk_header(FILE *f, const char id[4], uint64_t size) {
  return io_write_exact(f, id, 4) && io_write_u64(f, size);
}

static bool io_read_chunk_header(FILE *f, char id[4], uint64_t *size,
                                 bool *eof) {
  if (!f || !id || !size || !eof)
    return false;
  *eof = false;
  size_t n = fread(id, 1, 4, f);
  if (n == 0) {
    *eof = true;
    return true;
  }
  if (n != 4)
    return false;
  return io_read_u64(f, size);
}

static bool io_write_layer_desc(FILE *f, const NmSerLayerDesc *d) {
  if (!f || !d)
    return false;

  if (!io_write_u32(f, (uint32_t)d->type) ||
      !io_write_u32(f, (uint32_t)d->activation) ||
      !io_write_shape(f, d->input_shape) ||
      !io_write_shape(f, d->output_shape)) {
    return false;
  }

  switch (d->type) {
  case NM_LAYER_DENSE:
    return io_write_u64(f, (uint64_t)d->cfg.units);
  case NM_LAYER_CONV2D:
    return io_write_u64(f, (uint64_t)d->cfg.filters) &&
           io_write_u64(f, (uint64_t)d->cfg.kernel_size) &&
           io_write_u64(f, (uint64_t)d->cfg.stride) &&
           io_write_u64(f, (uint64_t)d->cfg.padding) &&
           io_write_u32(f, (uint32_t)d->cfg.algo) &&
           io_write_u32(f, (uint32_t)d->cfg.layout);
  case NM_LAYER_MAXPOOL2D:
    return io_write_u64(f, (uint64_t)d->cfg.kernel_h) &&
           io_write_u64(f, (uint64_t)d->cfg.kernel_w) &&
           io_write_u64(f, (uint64_t)d->cfg.stride_h) &&
           io_write_u64(f, (uint64_t)d->cfg.stride_w) &&
           io_write_u64(f, (uint64_t)d->cfg.pad_h) &&
           io_write_u64(f, (uint64_t)d->cfg.pad_w);
  case NM_LAYER_FLATTEN:
    return true;
  default:
    return false;
  }
}

static bool io_read_layer_desc(FILE *f, NmSerLayerDesc *d) {
  if (!f || !d)
    return false;
  uint32_t t = 0;
  uint32_t act = 0;
  if (!io_read_u32(f, &t) || !io_read_u32(f, &act) ||
      !io_read_shape(f, &d->input_shape) ||
      !io_read_shape(f, &d->output_shape)) {
    return false;
  }

  d->type = (NmLayerType)t;
  d->activation = (NmActivation)act;

  switch (d->type) {
  case NM_LAYER_DENSE: {
    uint64_t units = 0;
    if (!io_read_u64(f, &units))
      return false;
    d->cfg.units = (size_t)units;
    return true;
  }
  case NM_LAYER_CONV2D: {
    uint64_t filters = 0;
    uint64_t k = 0;
    uint64_t stride = 0;
    uint64_t pad = 0;
    uint32_t algo = 0;
    uint32_t layout = 0;
    if (!io_read_u64(f, &filters) || !io_read_u64(f, &k) ||
        !io_read_u64(f, &stride) || !io_read_u64(f, &pad) ||
        !io_read_u32(f, &algo) || !io_read_u32(f, &layout)) {
      return false;
    }
    d->cfg.filters = (size_t)filters;
    d->cfg.kernel_size = (size_t)k;
    d->cfg.stride = (size_t)stride;
    d->cfg.padding = (size_t)pad;
    d->cfg.algo = (NmConvAlgo)algo;
    d->cfg.layout = (NmTensorLayout)layout;
    return true;
  }
  case NM_LAYER_MAXPOOL2D: {
    uint64_t kh = 0, kw = 0, sh = 0, sw = 0, ph = 0, pw = 0;
    if (!io_read_u64(f, &kh) || !io_read_u64(f, &kw) || !io_read_u64(f, &sh) ||
        !io_read_u64(f, &sw) || !io_read_u64(f, &ph) || !io_read_u64(f, &pw)) {
      return false;
    }
    d->cfg.kernel_h = (size_t)kh;
    d->cfg.kernel_w = (size_t)kw;
    d->cfg.stride_h = (size_t)sh;
    d->cfg.stride_w = (size_t)sw;
    d->cfg.pad_h = (size_t)ph;
    d->cfg.pad_w = (size_t)pw;
    return true;
  }
  case NM_LAYER_FLATTEN:
    return true;
  default:
    return false;
  }
}

static uint64_t nm_seq_meta_chunk_size(const NmSeq *m, size_t tensor_count) {
  (void)tensor_count;
  if (!m)
    return 0;
  return sizeof(uint32_t) * 4 + io_string_size(m->runtime.backend_name) +
         io_string_size("nm.seq");
}

static uint64_t nm_seq_layer_desc_size_v3(const NmSerLayerDesc *d) {
  return io_string_size(d ? d->name : NULL) + nm_seq_layer_desc_size(d);
}

static uint64_t nm_seq_layers_chunk_size_v3(const NmSerLayerDesc *descs,
                                            size_t count) {
  uint64_t total = 0;
  if (!descs)
    return 0;
  for (size_t i = 0; i < count; ++i)
    total += nm_seq_layer_desc_size_v3(&descs[i]);
  return total;
}

static bool io_write_layer_desc_v3(FILE *f, const NmSerLayerDesc *d) {
  return io_write_string(f, d ? d->name : NULL) && io_write_layer_desc(f, d);
}

static bool io_read_layer_desc_v3(FILE *f, NmSerLayerDesc *d) {
  if (!d || !io_read_string(f, d->name, sizeof(d->name)))
    return false;
  return io_read_layer_desc(f, d);
}

static uint64_t nm_seq_tensor_desc_size(const NmSerTensorDesc *d) {
  return io_string_size(d->name) + io_string_size(d->layer_name) +
         sizeof(uint32_t) * 3 + sizeof(uint64_t) * 5 + sizeof(uint8_t);
}

static uint64_t nm_seq_tensors_chunk_size(const NmSerTensorDesc *descs,
                                          size_t count) {
  uint64_t total = sizeof(uint64_t);
  for (size_t i = 0; i < count; ++i)
    total += nm_seq_tensor_desc_size(&descs[i]);
  return total;
}

static bool io_write_tensor_desc(FILE *f, const NmSerTensorDesc *d) {
  if (!f || !d)
    return false;
  return io_write_string(f, d->name) && io_write_string(f, d->layer_name) &&
         io_write_u32(f, d->layer_index) && io_write_u32(f, d->role) &&
         io_write_u32(f, d->ndim) && io_write_u64(f, d->dims[0]) &&
         io_write_u64(f, d->dims[1]) && io_write_u64(f, d->dims[2]) &&
         io_write_u64(f, d->dims[3]) && io_write_u64(f, d->data_bytes) &&
         io_write_u8(f, d->dtype);
}

static bool io_read_tensor_desc(FILE *f, NmSerTensorDesc *d) {
  if (!f || !d)
    return false;
  memset(d, 0, sizeof(*d));
  return io_read_string(f, d->name, sizeof(d->name)) &&
         io_read_string(f, d->layer_name, sizeof(d->layer_name)) &&
         io_read_u32(f, &d->layer_index) && io_read_u32(f, &d->role) &&
         io_read_u32(f, &d->ndim) && io_read_u64(f, &d->dims[0]) &&
         io_read_u64(f, &d->dims[1]) && io_read_u64(f, &d->dims[2]) &&
         io_read_u64(f, &d->dims[3]) && io_read_u64(f, &d->data_bytes) &&
         io_read_u8(f, &d->dtype);
}

static bool io_read_tensor_desc_v3(FILE *f, NmSerTensorDesc *d) {
  if (!f || !d)
    return false;
  memset(d, 0, sizeof(*d));
  d->dtype = (uint8_t)ST_DTYPE_F32;
  return io_read_string(f, d->name, sizeof(d->name)) &&
         io_read_string(f, d->layer_name, sizeof(d->layer_name)) &&
         io_read_u32(f, &d->layer_index) && io_read_u32(f, &d->role) &&
         io_read_u32(f, &d->ndim) && io_read_u64(f, &d->dims[0]) &&
         io_read_u64(f, &d->dims[1]) && io_read_u64(f, &d->dims[2]) &&
         io_read_u64(f, &d->dims[3]) && io_read_u64(f, &d->data_bytes);
}

static bool checked_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
  if (!out || UINT64_MAX - a < b)
    return false;
  *out = a + b;
  return true;
}

static bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
  if (!out)
    return false;
  if (a != 0 && b > UINT64_MAX / a)
    return false;
  *out = a * b;
  return true;
}

static uint64_t nm_seq_min_layer_desc_size(bool legacy_format) {
  uint64_t size = sizeof(uint32_t) * 2 + sizeof(uint64_t) * 8;
  if (!legacy_format)
    size += sizeof(uint16_t);
  return size;
}

static uint64_t nm_seq_min_tensor_desc_size(bool with_dtype) {
  uint64_t size =
      sizeof(uint16_t) * 2 + sizeof(uint32_t) * 3 + sizeof(uint64_t) * 5;
  if (with_dtype)
    size += sizeof(uint8_t);
  return size;
}

static bool tensor_desc_file_dtype(const NmSerTensorDesc *descs, size_t count,
                                   StDtype *out_dtype) {
  if (!out_dtype)
    return false;
  if (!descs || count == 0) {
    *out_dtype = ST_DTYPE_F32;
    return true;
  }

  StDtype dtype = (StDtype)descs[0].dtype;
  if (dtype != ST_DTYPE_F32 && dtype != ST_DTYPE_BF16)
    return false;

  for (size_t i = 1; i < count; ++i) {
    if ((StDtype)descs[i].dtype != dtype)
      return false;
  }

  *out_dtype = dtype;
  return true;
}

static bool weight_stream_size_for_dtype(const NmSeq *m, StDtype dtype,
                                         uint64_t *out_size) {
  if (!m || !out_size)
    return false;

  if (dtype != ST_DTYPE_F32 && dtype != ST_DTYPE_BF16)
    return false;

  uint64_t elem_size = (uint64_t)st_dtype_size(dtype);
  uint64_t total = 0;
  for (size_t i = 0; i < m->num_layers; ++i) {
    const NmLayerEntry *e = &m->layers[i];
    if (!checked_add_u64(total, sizeof(uint32_t), &total))
      return false;

    if (e->type == NM_LAYER_DENSE) {
      const NmDenseParams *p = (const NmDenseParams *)e->params;
      uint64_t weight_elems = 0;
      uint64_t weight_bytes = 0;
      uint64_t bias_bytes = 0;
      if (!p || !p->weights || !p->bias ||
          !checked_mul_u64((uint64_t)p->weights->rows,
                           (uint64_t)p->weights->cols, &weight_elems) ||
          !checked_mul_u64(weight_elems, elem_size, &weight_bytes) ||
          !checked_mul_u64((uint64_t)p->bias->cols, elem_size, &bias_bytes) ||
          !checked_add_u64(total, sizeof(uint64_t) * 3, &total) ||
          !checked_add_u64(total, weight_bytes, &total) ||
          !checked_add_u64(total, bias_bytes, &total)) {
        return false;
      }
    } else if (e->type == NM_LAYER_CONV2D) {
      const NmConv2dInternalParams *p =
          (const NmConv2dInternalParams *)e->params;
      uint64_t weight_bytes = 0;
      uint64_t bias_bytes = 0;
      if (!p || !p->weight || !p->bias ||
          !checked_mul_u64((uint64_t)p->weight->numel, elem_size,
                           &weight_bytes) ||
          !checked_mul_u64((uint64_t)p->bias->numel, elem_size, &bias_bytes) ||
          !checked_add_u64(total, sizeof(uint64_t) * 6, &total) ||
          !checked_add_u64(total, weight_bytes, &total) ||
          !checked_add_u64(total, bias_bytes, &total)) {
        return false;
      }
    }
  }

  *out_size = total;
  return true;
}

static uint64_t nm_seq_plan_chunk_size(void) { return 32ULL; }

static uint64_t nm_seq_training_chunk_size(void) {
  return sizeof(uint32_t) * 2 + sizeof(float) * 5 + sizeof(uint64_t) * 2 + 2ULL;
}

static NmSeq *build_model_from_ser_desc(const NmSerModelDesc *desc) {
  if (!desc)
    return NULL;

  NmSeq *m = nm_seq_create(desc->input_shape);
  if (!m)
    return NULL;

  if (nm_seq_set_tensor_layout(m, desc->layout) != NM_STATUS_OK ||
      nm_seq_set_optimizer(m, &desc->optimizer_config) != NM_STATUS_OK) {
    nm_seq_free(m);
    return NULL;
  }

  for (size_t i = 0; i < desc->layer_count; ++i) {
    const NmSerLayerDesc *d = &desc->layers[i];
    NmStatus status = NM_STATUS_OK;
    switch (d->type) {
    case NM_LAYER_DENSE: {
      NmDenseConfig cfg = {.units = d->cfg.units, .activation = d->activation};
      status = nm_seq_add_dense(m, &cfg);
      break;
    }
    case NM_LAYER_CONV2D: {
      NmConv2dConfig cfg = {
          .filters = d->cfg.filters,
          .kernel_size = d->cfg.kernel_size,
          .stride = d->cfg.stride,
          .padding = d->cfg.padding,
          .algo = d->cfg.algo,
          .layout = d->cfg.layout,
          .activation = d->activation,
      };
      status = nm_seq_add_conv2d(m, &cfg);
      break;
    }
    case NM_LAYER_MAXPOOL2D: {
      NmPool2dConfig cfg = {
          .kernel_size = d->cfg.kernel_h,
          .stride = d->cfg.stride_h,
          .padding = d->cfg.pad_h,
      };
      status = nm_seq_add_maxpool2d(m, &cfg);
      break;
    }
    case NM_LAYER_FLATTEN:
      status = nm_seq_add_flatten(m);
      break;
    default:
      status = NM_STATUS_UNSUPPORTED_TYPE;
      break;
    }

    if (status != NM_STATUS_OK) {
      nm_seq_free(m);
      return NULL;
    }
  }

  if (nm_seq_compile(m, &desc->compile_config) != NM_STATUS_OK) {
    nm_seq_free(m);
    return NULL;
  }
  return m;
}

static bool write_weight_data(FILE *f, const float *data, size_t numel,
                              StDtype dtype) {
  if (dtype == ST_DTYPE_BF16) {
    uint16_t *tmp = (uint16_t *)malloc(numel * sizeof(uint16_t));
    if (!tmp)
      return false;
    st_f32_to_bf16_bulk(data, tmp, numel);
    bool ok = io_write_exact(f, tmp, numel * sizeof(uint16_t));
    free(tmp);
    return ok;
  }
  return io_write_exact(f, data, numel * sizeof(float));
}

static bool read_weight_data(FILE *f, float *data, size_t numel,
                             StDtype file_dtype) {
  if (file_dtype == ST_DTYPE_BF16) {
    uint16_t *tmp = (uint16_t *)malloc(numel * sizeof(uint16_t));
    if (!tmp)
      return false;
    bool ok = io_read_exact(f, tmp, numel * sizeof(uint16_t));
    if (ok)
      st_bf16_to_f32_bulk(tmp, data, numel);
    free(tmp);
    return ok;
  }
  return io_read_exact(f, data, numel * sizeof(float));
}

static NmStatus nm_write_weights_to_ex(const NmSeq *m, FILE *f,
                                       StDtype export_dtype) {
  for (size_t i = 0; i < m->num_layers; ++i) {
    const NmLayerEntry *e = &m->layers[i];
    if (!io_write_u32(f, (uint32_t)e->type))
      return NM_STATUS_IO_ERROR;
    if (e->type == NM_LAYER_DENSE) {
      const NmDenseParams *p = (const NmDenseParams *)e->params;
      uint64_t wr = (uint64_t)p->weights->rows;
      uint64_t wc = (uint64_t)p->weights->cols;
      if (!io_write_u64(f, wr) || !io_write_u64(f, wc) ||
          !write_weight_data(f, p->weights->values, (size_t)(wr * wc),
                             export_dtype) ||
          !io_write_u64(f, (uint64_t)p->bias->cols) ||
          !write_weight_data(f, p->bias->values, p->bias->cols, export_dtype)) {
        return NM_STATUS_IO_ERROR;
      }
    } else if (e->type == NM_LAYER_CONV2D) {
      const NmConv2dInternalParams *p =
          (const NmConv2dInternalParams *)e->params;
      if (!io_write_u64(f, (uint64_t)p->weight->ndim) ||
          !io_write_u64(f, (uint64_t)p->weight->shape[0]) ||
          !io_write_u64(f, (uint64_t)p->weight->shape[1]) ||
          !io_write_u64(f, (uint64_t)p->weight->shape[2]) ||
          !io_write_u64(f, (uint64_t)p->weight->shape[3]) ||
          !write_weight_data(f, p->weight->values, p->weight->numel,
                             export_dtype) ||
          !io_write_u64(f, (uint64_t)p->bias->numel) ||
          !write_weight_data(f, p->bias->values, p->bias->numel,
                             export_dtype)) {
        return NM_STATUS_IO_ERROR;
      }
    }
  }
  return NM_STATUS_OK;
}

static NmStatus nm_load_weights_from_ex(NmSeq *m, FILE *f, StDtype file_dtype) {
  for (size_t i = 0; i < m->num_layers; ++i) {
    uint32_t lt = 0;
    if (!io_read_u32(f, &lt))
      return NM_STATUS_IO_ERROR;
    if ((NmLayerType)lt != m->layers[i].type)
      return NM_STATUS_FORMAT_ERROR;

    NmLayerEntry *e = &m->layers[i];
    if (e->type == NM_LAYER_DENSE) {
      NmDenseParams *p = (NmDenseParams *)e->params;
      uint64_t wr = 0;
      uint64_t wc = 0;
      uint64_t bc = 0;
      if (!io_read_u64(f, &wr) || !io_read_u64(f, &wc))
        return NM_STATUS_IO_ERROR;
      if (wr != (uint64_t)p->weights->rows || wc != (uint64_t)p->weights->cols)
        return NM_STATUS_FORMAT_ERROR;
      if (!read_weight_data(f, p->weights->values, (size_t)(wr * wc),
                            file_dtype) ||
          !io_read_u64(f, &bc) || bc != (uint64_t)p->bias->cols ||
          !read_weight_data(f, p->bias->values, p->bias->cols, file_dtype)) {
        return NM_STATUS_IO_ERROR;
      }
    } else if (e->type == NM_LAYER_CONV2D) {
      NmConv2dInternalParams *p = (NmConv2dInternalParams *)e->params;
      uint64_t ndim = 0, s0 = 0, s1 = 0, s2 = 0, s3 = 0, bn = 0;
      if (!io_read_u64(f, &ndim) || !io_read_u64(f, &s0) ||
          !io_read_u64(f, &s1) || !io_read_u64(f, &s2) ||
          !io_read_u64(f, &s3)) {
        return NM_STATUS_IO_ERROR;
      }
      if (ndim != 4 || s0 != (uint64_t)p->weight->shape[0] ||
          s1 != (uint64_t)p->weight->shape[1] ||
          s2 != (uint64_t)p->weight->shape[2] ||
          s3 != (uint64_t)p->weight->shape[3]) {
        return NM_STATUS_FORMAT_ERROR;
      }
      if (!read_weight_data(f, p->weight->values, p->weight->numel,
                            file_dtype) ||
          !io_read_u64(f, &bn) || bn != (uint64_t)p->bias->numel ||
          !read_weight_data(f, p->bias->values, p->bias->numel, file_dtype)) {
        return NM_STATUS_IO_ERROR;
      }
    }
  }
  return NM_STATUS_OK;
}

static NmStatus nm_seq_save_full(const NmSeq *m, const char *path,
                                 bool inference_only, StDtype export_dtype) {
  if (!m || !path || m->num_layers == 0 || !m->layers)
    return NM_STATUS_INVALID_ARGUMENT;

  FILE *f = fopen(path, "w+b");
  if (!f)
    return NM_STATUS_IO_ERROR;

  NmStatus status = NM_STATUS_OK;
  NmSerModelDesc ser_desc;
  ser_model_desc_init(&ser_desc);
  status =
      ser_model_desc_from_model(m, inference_only, export_dtype, &ser_desc);
  if (status != NM_STATUS_OK) {
    fclose(f);
    return status;
  }

  if (!io_write_exact(f, NM_SEQ_BIN_MAGIC, 4) ||
      !io_write_u32(f, NM_SEQ_BIN_VERSION)) {
    status = NM_STATUS_IO_ERROR;
    goto done;
  }

  {
    uint64_t raw_size = nm_seq_meta_chunk_size(m, ser_desc.tensor_count);
    uint64_t padded_size = align_u64(raw_size, NM_SEQ_ALIGNMENT);
    if (!io_write_chunk_header(f, NM_SEQ_CHUNK_META, padded_size) ||
        !io_write_u32(f, ser_desc.alignment) ||
        !io_write_u32(f, ser_desc.flags) ||
        !io_write_u32(f, ser_desc.format_version) ||
        !io_write_u32(f, (uint32_t)ser_desc.tensor_count) ||
        !io_write_string(f, ser_desc.backend_name) ||
        !io_write_string(f, ser_desc.architecture) ||
        !io_write_padding(f, padded_size - raw_size)) {
      status = NM_STATUS_IO_ERROR;
      goto done;
    }
  }

  {
    uint64_t raw_size = nm_seq_conf_chunk_size();
    uint64_t padded_size = align_u64(raw_size, NM_SEQ_ALIGNMENT);
    if (!io_write_chunk_header(f, NM_SEQ_CHUNK_CONF, padded_size) ||
        !io_write_shape(f, ser_desc.input_shape) ||
        !io_write_u32(f, (uint32_t)ser_desc.compile_config.loss_type) ||
        !io_write_u8(f, (uint8_t)(compile_uses_softmax_ce_shortcut(
                                      m, ser_desc.compile_config.loss_type)
                                      ? 1
                                      : 0)) ||
        !io_write_u32(f, (uint32_t)ser_desc.optimizer_config.type) ||
        !io_write_f32(f, ser_desc.optimizer_config.learning_rate) ||
        !io_write_f32(f, ser_desc.optimizer_config.momentum) ||
        !io_write_f32(f, ser_desc.optimizer_config.beta1) ||
        !io_write_f32(f, ser_desc.optimizer_config.beta2) ||
        !io_write_f32(f, ser_desc.optimizer_config.epsilon) ||
        !io_write_u32(f, (uint32_t)ser_desc.layout) ||
        !io_write_u64(f, (uint64_t)ser_desc.layer_count) ||
        !io_write_u32(f, (uint32_t)ser_desc.autograd_mode) ||
        !io_write_u8(f, (uint8_t)(ser_desc.batch_parallel.enabled ? 1 : 0)) ||
        !io_write_u8(
            f, (uint8_t)(ser_desc.batch_parallel.deterministic ? 1 : 0)) ||
        !io_write_u64(f, (uint64_t)ser_desc.batch_parallel.num_workers) ||
        !io_write_padding(f, padded_size - raw_size)) {
      status = NM_STATUS_IO_ERROR;
      goto done;
    }
  }

  {
    uint64_t raw_size =
        nm_seq_layers_chunk_size_v3(ser_desc.layers, ser_desc.layer_count);
    uint64_t padded_size = align_u64(raw_size, NM_SEQ_ALIGNMENT);
    if (!io_write_chunk_header(f, NM_SEQ_CHUNK_LAYS, padded_size)) {
      status = NM_STATUS_IO_ERROR;
      goto done;
    }
    for (size_t i = 0; i < ser_desc.layer_count; ++i) {
      if (!io_write_layer_desc_v3(f, &ser_desc.layers[i])) {
        status = NM_STATUS_IO_ERROR;
        goto done;
      }
    }
    if (!io_write_padding(f, padded_size - raw_size)) {
      status = NM_STATUS_IO_ERROR;
      goto done;
    }
  }

  {
    uint64_t raw_size =
        nm_seq_tensors_chunk_size(ser_desc.tensors, ser_desc.tensor_count);
    uint64_t padded_size = align_u64(raw_size, NM_SEQ_ALIGNMENT);
    if (!io_write_chunk_header(f, NM_SEQ_CHUNK_TENS, padded_size) ||
        !io_write_u64(f, (uint64_t)ser_desc.tensor_count)) {
      status = NM_STATUS_IO_ERROR;
      goto done;
    }
    for (size_t i = 0; i < ser_desc.tensor_count; ++i) {
      if (!io_write_tensor_desc(f, &ser_desc.tensors[i])) {
        status = NM_STATUS_IO_ERROR;
        goto done;
      }
    }
    if (!io_write_padding(f, padded_size - raw_size)) {
      status = NM_STATUS_IO_ERROR;
      goto done;
    }
  }

  {
    char *wraw = NULL;
    size_t wraw_size = 0;
    FILE *wf = open_memstream(&wraw, &wraw_size);
    if (!wf) {
      status = NM_STATUS_IO_ERROR;
      goto done;
    }
    status = nm_write_weights_to_ex(m, wf, export_dtype);
    fclose(wf);
    if (status != NM_STATUS_OK || !wraw) {
      free(wraw);
      if (status == NM_STATUS_OK)
        status = NM_STATUS_IO_ERROR;
      goto done;
    }
    size_t zbound = ZSTD_compressBound(wraw_size);
    void *zbuf = malloc(zbound);
    if (!zbuf) {
      free(wraw);
      status = NM_STATUS_ALLOC_FAILED;
      goto done;
    }
    size_t zsize = ZSTD_compress(zbuf, zbound, wraw, wraw_size, 3);
    free(wraw);
    uint64_t raw_size = sizeof(uint64_t) * 2 + (uint64_t)zsize;
    uint64_t padded_size = align_u64(raw_size, NM_SEQ_ALIGNMENT);
    if (ZSTD_isError(zsize) ||
        !io_write_chunk_header(f, NM_SEQ_CHUNK_WGHZ, padded_size) ||
        !io_write_u64(f, (uint64_t)wraw_size) ||
        !io_write_u64(f, (uint64_t)zsize) || !io_write_exact(f, zbuf, zsize) ||
        !io_write_padding(f, padded_size - raw_size)) {
      free(zbuf);
      status = NM_STATUS_IO_ERROR;
      goto done;
    }
    free(zbuf);
  }

  {
    uint64_t raw_size = nm_seq_plan_chunk_size();
    uint64_t padded_size = align_u64(raw_size, NM_SEQ_ALIGNMENT);
    if (!io_write_chunk_header(f, NM_SEQ_CHUNK_PLAN, padded_size) ||
        !io_write_u64(f, ser_desc.plan_batch_rows) ||
        !io_write_u64(f, ser_desc.plan_output_cols) ||
        !io_write_u64(f, ser_desc.plan_peak_tensors) ||
        !io_write_u64(f, ser_desc.plan_peak_bytes) ||
        !io_write_padding(f, padded_size - raw_size)) {
      status = NM_STATUS_IO_ERROR;
      goto done;
    }
  }

  if (!inference_only) {
    uint64_t raw_size = nm_seq_training_chunk_size();
    uint64_t padded_size = align_u64(raw_size, NM_SEQ_ALIGNMENT);
    if (!io_write_chunk_header(f, NM_SEQ_CHUNK_TRNG, padded_size) ||
        !io_write_u32(f, (uint32_t)ser_desc.optimizer_config.type) ||
        !io_write_f32(f, ser_desc.optimizer_config.learning_rate) ||
        !io_write_f32(f, ser_desc.optimizer_config.momentum) ||
        !io_write_f32(f, ser_desc.optimizer_config.beta1) ||
        !io_write_f32(f, ser_desc.optimizer_config.beta2) ||
        !io_write_f32(f, ser_desc.optimizer_config.epsilon) ||
        !io_write_u64(f, ser_desc.optimizer_step) ||
        !io_write_u32(f, (uint32_t)ser_desc.autograd_mode) ||
        !io_write_u8(f, (uint8_t)(ser_desc.batch_parallel.enabled ? 1 : 0)) ||
        !io_write_u8(
            f, (uint8_t)(ser_desc.batch_parallel.deterministic ? 1 : 0)) ||
        !io_write_u64(f, (uint64_t)ser_desc.batch_parallel.num_workers) ||
        !io_write_padding(f, padded_size - raw_size)) {
      status = NM_STATUS_IO_ERROR;
      goto done;
    }
  }

  {
    long content_end = ftell(f);
    if (content_end < 0) {
      status = NM_STATUS_IO_ERROR;
      goto done;
    }
    if (fflush(f) != 0 || fseek(f, 8, SEEK_SET) != 0) {
      status = NM_STATUS_IO_ERROR;
      goto done;
    }
    uint8_t hbuf[4096];
    XXH64_state_t *hstate = XXH64_createState();
    if (!hstate) {
      status = NM_STATUS_ALLOC_FAILED;
      goto done;
    }
    XXH64_reset(hstate, 0);
    long to_hash = content_end - 8;
    while (to_hash > 0) {
      size_t step =
          to_hash < (long)sizeof(hbuf) ? (size_t)to_hash : sizeof(hbuf);
      size_t n = fread(hbuf, 1, step, f);
      if (n > 0)
        XXH64_update(hstate, hbuf, n);
      to_hash -= (long)step;
    }
    uint64_t digest = (uint64_t)XXH64_digest(hstate);
    XXH64_freeState(hstate);
    if (fseek(f, content_end, SEEK_SET) != 0 ||
        !io_write_chunk_header(f, NM_SEQ_CHUNK_HASH,
                               align_u64(sizeof(uint64_t), NM_SEQ_ALIGNMENT)) ||
        !io_write_u64(f, digest) ||
        !io_write_padding(f, align_u64(sizeof(uint64_t), NM_SEQ_ALIGNMENT) -
                                 sizeof(uint64_t))) {
      status = NM_STATUS_IO_ERROR;
      goto done;
    }
  }

done:
  ser_model_desc_destroy(&ser_desc);
  if (fclose(f) != 0 && status == NM_STATUS_OK)
    status = NM_STATUS_IO_ERROR;
  return status;
}

NmStatus nm_seq_save(const NmSeq *m, const char *path) {
  return nm_seq_save_full(m, path, false, ST_DTYPE_F32);
}

NmStatus nm_seq_save_inference(const NmSeq *m, const char *path) {
  return nm_seq_save_full(m, path, true, ST_DTYPE_F32);
}

NmStatus nm_seq_save_inference_ex(const NmSeq *m, const char *path,
                                  const NmSaveInferenceOptions *opts) {
  StDtype dtype = (opts && opts->use_bf16) ? ST_DTYPE_BF16 : ST_DTYPE_F32;
  return nm_seq_save_full(m, path, true, dtype);
}

NmStatus nm_seq_load(const char *path, NmSeq **out_model) {
  if (!path || !out_model)
    return NM_STATUS_INVALID_ARGUMENT;

  *out_model = NULL;

  FILE *f = fopen(path, "rb");
  if (!f)
    return NM_STATUS_IO_ERROR;

  NmStatus status = NM_STATUS_OK;
  char magic[4] = {0};
  uint32_t version = 0;
  bool legacy_format = false;
  if (!io_read_exact(f, magic, 4) || !io_read_u32(f, &version)) {
    status = NM_STATUS_IO_ERROR;
    goto done;
  }

  if (memcmp(magic, NM_SEQ_BIN_MAGIC, 4) == 0 &&
      (version == NM_SEQ_BIN_VERSION || version == NM_SEQ_BIN_VERSION_V3)) {
    legacy_format = false;
  } else if (memcmp(magic, NM_SEQ_BIN_MAGIC_LEGACY, 4) == 0 &&
             version == NM_SEQ_BIN_VERSION_LEGACY) {
    legacy_format = true;
  } else {
    status = NM_STATUS_FORMAT_ERROR;
    goto done;
  }

  bool has_conf = false;
  bool has_lays = false;
  bool has_wght = false;
  NmShape input_shape = {0};
  NmCompileConfig compile_cfg = nm_compile_defaults();
  NmOptimizerConfig opt_cfg = optimizer_defaults_internal();
  NmTensorLayout layout = NM_LAYOUT_NCHW;
  NmAutogradMode autograd_mode = NM_AUTOGRAD_DISABLED;
  NmBatchParallelConfig batch_cfg = {
      .enabled = false,
      .deterministic = true,
      .num_workers = 0,
  };
  uint32_t meta_alignment = NM_SEQ_ALIGNMENT;
  uint32_t meta_flags = 0;
  uint64_t plan_batch_rows = 0;
  uint64_t plan_output_cols = 0;
  uint64_t plan_peak_tensors = 0;
  uint64_t plan_peak_bytes = 0;
  uint64_t num_layers_u64 = 0;
  NmSerLayerDesc *descs = NULL;
  NmSerTensorDesc *tensor_descs = NULL;
  uint64_t tensor_desc_count = 0;
  NmSerModelDesc build_desc;
  ser_model_desc_init(&build_desc);
  NmSeq *loaded = NULL;
  uint64_t loaded_step = 0;

  while (true) {
    char id[4] = {0};
    uint64_t chunk_size = 0;
    bool eof = false;
    if (!io_read_chunk_header(f, id, &chunk_size, &eof)) {
      status = NM_STATUS_IO_ERROR;
      goto fail;
    }
    if (eof)
      break;

    long chunk_start = ftell(f);
    if (chunk_start < 0) {
      status = NM_STATUS_IO_ERROR;
      goto fail;
    }

    if (memcmp(id, NM_SEQ_CHUNK_META, 4) == 0) {
      uint32_t stored_version = 0;
      uint32_t tensor_count_u32 = 0;
      char backend_name[64];
      char architecture[32];
      if (!io_read_u32(f, &meta_alignment) || !io_read_u32(f, &meta_flags) ||
          !io_read_u32(f, &stored_version) ||
          !io_read_u32(f, &tensor_count_u32) ||
          !io_read_string(f, backend_name, sizeof(backend_name)) ||
          !io_read_string(f, architecture, sizeof(architecture))) {
        status = NM_STATUS_IO_ERROR;
        goto fail;
      }
      tensor_desc_count = tensor_count_u32;
    } else if (memcmp(id, NM_SEQ_CHUNK_CONF, 4) == 0) {
      uint32_t loss = 0;
      uint32_t opt_type = 0;
      uint32_t layout_u32 = 0;
      uint8_t shortcut = 0;
      if (!io_read_shape(f, &input_shape) || !io_read_u32(f, &loss) ||
          !io_read_u8(f, &shortcut) || !io_read_u32(f, &opt_type) ||
          !io_read_f32(f, &opt_cfg.learning_rate) ||
          !io_read_f32(f, &opt_cfg.momentum) ||
          !io_read_f32(f, &opt_cfg.beta1) || !io_read_f32(f, &opt_cfg.beta2) ||
          !io_read_f32(f, &opt_cfg.epsilon) || !io_read_u32(f, &layout_u32) ||
          !io_read_u64(f, &num_layers_u64)) {
        status = NM_STATUS_IO_ERROR;
        goto fail;
      }

      long conf_pos = ftell(f);
      if (conf_pos < 0) {
        status = NM_STATUS_IO_ERROR;
        goto fail;
      }
      uint64_t conf_consumed = (uint64_t)(conf_pos - chunk_start);
      uint64_t conf_remaining =
          chunk_size > conf_consumed ? (chunk_size - conf_consumed) : 0;

      if (conf_remaining >= 4) {
        uint32_t ag = 0;
        if (!io_read_u32(f, &ag)) {
          status = NM_STATUS_IO_ERROR;
          goto fail;
        }
        autograd_mode = (NmAutogradMode)ag;
        conf_remaining -= 4;
      }
      if (conf_remaining >= 1) {
        uint8_t en = 0;
        if (!io_read_u8(f, &en)) {
          status = NM_STATUS_IO_ERROR;
          goto fail;
        }
        batch_cfg.enabled = en != 0;
        conf_remaining -= 1;
      }
      if (conf_remaining >= 1) {
        uint8_t det = 1;
        if (!io_read_u8(f, &det)) {
          status = NM_STATUS_IO_ERROR;
          goto fail;
        }
        batch_cfg.deterministic = det != 0;
        conf_remaining -= 1;
      }
      if (conf_remaining >= 8) {
        uint64_t workers = 0;
        if (!io_read_u64(f, &workers)) {
          status = NM_STATUS_IO_ERROR;
          goto fail;
        }
        batch_cfg.num_workers = (size_t)workers;
      }

      compile_cfg.loss_type = (NmLoss)loss;
      (void)shortcut;
      opt_cfg.type = (NmOptimizerType)opt_type;
      layout = (NmTensorLayout)layout_u32;
      has_conf = true;
    } else if (memcmp(id, NM_SEQ_CHUNK_LAYS, 4) == 0) {
      if (!has_conf || num_layers_u64 == 0 ||
          num_layers_u64 > (uint64_t)SIZE_MAX) {
        status = NM_STATUS_FORMAT_ERROR;
        goto fail;
      }
      size_t n = (size_t)num_layers_u64;
      uint64_t min_desc_size = nm_seq_min_layer_desc_size(legacy_format);
      if (n > SIZE_MAX / sizeof(*descs) ||
          (uint64_t)n > chunk_size / min_desc_size) {
        status = NM_STATUS_FORMAT_ERROR;
        goto fail;
      }
      descs = (NmSerLayerDesc *)calloc(n, sizeof(*descs));
      if (!descs) {
        status = NM_STATUS_ALLOC_FAILED;
        goto fail;
      }
      for (size_t i = 0; i < n; ++i) {
        bool ok = legacy_format ? io_read_layer_desc(f, &descs[i])
                                : io_read_layer_desc_v3(f, &descs[i]);
        if (!ok) {
          status = NM_STATUS_FORMAT_ERROR;
          goto fail;
        }
      }
      has_lays = true;
    } else if (memcmp(id, NM_SEQ_CHUNK_TENS, 4) == 0) {
      uint64_t count = 0;
      if (!io_read_u64(f, &count) || count > (uint64_t)SIZE_MAX) {
        status = NM_STATUS_FORMAT_ERROR;
        goto fail;
      }
      long tens_pos = ftell(f);
      if (tens_pos < 0) {
        status = NM_STATUS_IO_ERROR;
        goto fail;
      }
      uint64_t tens_consumed = (uint64_t)(tens_pos - chunk_start);
      uint64_t tens_remaining =
          chunk_size > tens_consumed ? (chunk_size - tens_consumed) : 0;
      uint64_t min_desc_size =
          nm_seq_min_tensor_desc_size(version >= NM_SEQ_BIN_VERSION);
      if ((size_t)count > SIZE_MAX / sizeof(*tensor_descs) ||
          count > tens_remaining / min_desc_size) {
        status = NM_STATUS_FORMAT_ERROR;
        goto fail;
      }
      free(tensor_descs);
      tensor_descs = calloc((size_t)count, sizeof(*tensor_descs));
      if (count > 0 && !tensor_descs) {
        status = NM_STATUS_ALLOC_FAILED;
        goto fail;
      }
      tensor_desc_count = count;
      for (size_t i = 0; i < (size_t)count; ++i) {
        bool ok = (version >= NM_SEQ_BIN_VERSION)
                      ? io_read_tensor_desc(f, &tensor_descs[i])
                      : io_read_tensor_desc_v3(f, &tensor_descs[i]);
        if (!ok) {
          status = NM_STATUS_FORMAT_ERROR;
          goto fail;
        }
      }
      if (!tensor_desc_file_dtype(tensor_descs, (size_t)count, &(StDtype){0})) {
        status = NM_STATUS_FORMAT_ERROR;
        goto fail;
      }
    } else if (memcmp(id, NM_SEQ_CHUNK_WGHT, 4) == 0) {
      if (!has_conf || !has_lays || !descs) {
        status = NM_STATUS_FORMAT_ERROR;
        goto fail;
      }
      if (!loaded) {
        build_desc.input_shape = input_shape;
        build_desc.compile_config = compile_cfg;
        build_desc.optimizer_config = opt_cfg;
        build_desc.layout = layout;
        build_desc.layers = descs;
        build_desc.layer_count = (size_t)num_layers_u64;
        loaded = build_model_from_ser_desc(&build_desc);
        if (!loaded) {
          status = NM_STATUS_FORMAT_ERROR;
          goto fail;
        }
      }
      {
        StDtype wdtype = ST_DTYPE_F32;
        if (!tensor_desc_file_dtype(tensor_descs, (size_t)tensor_desc_count,
                                    &wdtype)) {
          status = NM_STATUS_FORMAT_ERROR;
          goto fail;
        }
        status = nm_load_weights_from_ex(loaded, f, wdtype);
      }
      if (status != NM_STATUS_OK)
        goto fail;
      has_wght = true;
    } else if (memcmp(id, NM_SEQ_CHUNK_WGHZ, 4) == 0) {
      if (!has_conf || !has_lays || !descs) {
        status = NM_STATUS_FORMAT_ERROR;
        goto fail;
      }
      if (!loaded) {
        build_desc.input_shape = input_shape;
        build_desc.compile_config = compile_cfg;
        build_desc.optimizer_config = opt_cfg;
        build_desc.layout = layout;
        build_desc.layers = descs;
        build_desc.layer_count = (size_t)num_layers_u64;
        loaded = build_model_from_ser_desc(&build_desc);
        if (!loaded) {
          status = NM_STATUS_FORMAT_ERROR;
          goto fail;
        }
      }
      uint64_t raw_size = 0;
      uint64_t zsize_u64 = 0;
      if (!io_read_u64(f, &raw_size)) {
        status = NM_STATUS_IO_ERROR;
        goto fail;
      }
      uint64_t csize = 0;
      if (!legacy_format) {
        if (!io_read_u64(f, &zsize_u64)) {
          status = NM_STATUS_IO_ERROR;
          goto fail;
        }
        csize = zsize_u64;
      } else {
        csize = chunk_size > (uint64_t)sizeof(uint64_t)
                    ? chunk_size - (uint64_t)sizeof(uint64_t)
                    : 0;
      }
      long wghz_pos = ftell(f);
      if (wghz_pos < 0) {
        status = NM_STATUS_IO_ERROR;
        goto fail;
      }
      uint64_t wghz_consumed = (uint64_t)(wghz_pos - chunk_start);
      uint64_t wghz_remaining =
          chunk_size > wghz_consumed ? (chunk_size - wghz_consumed) : 0;
      if (csize > wghz_remaining) {
        status = NM_STATUS_FORMAT_ERROR;
        goto fail;
      }
      StDtype wdtype = ST_DTYPE_F32;
      if (!tensor_desc_file_dtype(tensor_descs, (size_t)tensor_desc_count,
                                  &wdtype)) {
        status = NM_STATUS_FORMAT_ERROR;
        goto fail;
      }
      uint64_t expected_raw_size = 0;
      if (!weight_stream_size_for_dtype(loaded, wdtype, &expected_raw_size) ||
          raw_size != expected_raw_size ||
          csize > ZSTD_compressBound((size_t)expected_raw_size)) {
        status = NM_STATUS_FORMAT_ERROR;
        goto fail;
      }
      void *zbuf = malloc((size_t)csize);
      if (!zbuf) {
        status = NM_STATUS_ALLOC_FAILED;
        goto fail;
      }
      if (fread(zbuf, 1, (size_t)csize, f) != (size_t)csize) {
        free(zbuf);
        status = NM_STATUS_IO_ERROR;
        goto fail;
      }
      void *raw = malloc((size_t)raw_size);
      if (!raw) {
        free(zbuf);
        status = NM_STATUS_ALLOC_FAILED;
        goto fail;
      }
      size_t res = ZSTD_decompress(raw, (size_t)raw_size, zbuf, (size_t)csize);
      free(zbuf);
      if (ZSTD_isError(res) || res != (size_t)expected_raw_size) {
        free(raw);
        status = NM_STATUS_FORMAT_ERROR;
        goto fail;
      }
      FILE *mf = fmemopen(raw, (size_t)raw_size, "rb");
      if (!mf) {
        free(raw);
        status = NM_STATUS_IO_ERROR;
        goto fail;
      }
      {
        status = nm_load_weights_from_ex(loaded, mf, wdtype);
      }
      fclose(mf);
      free(raw);
      if (status != NM_STATUS_OK)
        goto fail;
      has_wght = true;
    } else if (memcmp(id, NM_SEQ_CHUNK_HASH, 4) == 0) {
      uint64_t stored = 0;
      if (!io_read_u64(f, &stored)) {
        status = NM_STATUS_IO_ERROR;
        goto fail;
      }
      long after_hash_data = ftell(f);
      long hash_hdr_start = chunk_start - 12;
      if (after_hash_data < 0 || hash_hdr_start < 8) {
        status = NM_STATUS_FORMAT_ERROR;
        goto fail;
      }
      uint8_t hbuf[4096];
      XXH64_state_t *hstate = XXH64_createState();
      if (!hstate) {
        status = NM_STATUS_ALLOC_FAILED;
        goto fail;
      }
      XXH64_reset(hstate, 0);
      if (fseek(f, 8, SEEK_SET) != 0) {
        XXH64_freeState(hstate);
        status = NM_STATUS_IO_ERROR;
        goto fail;
      }
      long to_hash = hash_hdr_start - 8;
      while (to_hash > 0) {
        size_t step =
            to_hash < (long)sizeof(hbuf) ? (size_t)to_hash : sizeof(hbuf);
        size_t n = fread(hbuf, 1, step, f);
        if (n > 0)
          XXH64_update(hstate, hbuf, n);
        to_hash -= (long)step;
      }
      uint64_t computed = (uint64_t)XXH64_digest(hstate);
      XXH64_freeState(hstate);
      if (fseek(f, after_hash_data, SEEK_SET) != 0) {
        status = NM_STATUS_IO_ERROR;
        goto fail;
      }
      if (computed != stored) {
        log_error("Model integrity check failed (checksum mismatch)");
        status = NM_STATUS_FORMAT_ERROR;
        goto fail;
      }
    } else if (memcmp(id, NM_SEQ_CHUNK_PLAN, 4) == 0) {
      if (!io_read_u64(f, &plan_batch_rows) ||
          !io_read_u64(f, &plan_output_cols) ||
          !io_read_u64(f, &plan_peak_tensors) ||
          !io_read_u64(f, &plan_peak_bytes)) {
        status = NM_STATUS_IO_ERROR;
        goto fail;
      }
    } else if (memcmp(id, NM_SEQ_CHUNK_TRNG, 4) == 0) {
      uint32_t opt_type = 0;
      uint32_t ag = 0;
      uint8_t en = 0;
      uint8_t det = 1;
      uint64_t workers = 0;
      if (!io_read_u32(f, &opt_type) ||
          !io_read_f32(f, &opt_cfg.learning_rate) ||
          !io_read_f32(f, &opt_cfg.momentum) ||
          !io_read_f32(f, &opt_cfg.beta1) || !io_read_f32(f, &opt_cfg.beta2) ||
          !io_read_f32(f, &opt_cfg.epsilon) || !io_read_u64(f, &loaded_step) ||
          !io_read_u32(f, &ag) || !io_read_u8(f, &en) || !io_read_u8(f, &det) ||
          !io_read_u64(f, &workers)) {
        status = NM_STATUS_IO_ERROR;
        goto fail;
      }
      opt_cfg.type = (NmOptimizerType)opt_type;
      autograd_mode = (NmAutogradMode)ag;
      batch_cfg.enabled = en != 0;
      batch_cfg.deterministic = det != 0;
      batch_cfg.num_workers = (size_t)workers;
    } else if (memcmp(id, NM_SEQ_CHUNK_OPTS, 4) == 0) {
      if (!io_read_u64(f, &loaded_step)) {
        status = NM_STATUS_IO_ERROR;
        goto fail;
      }
    }

    long chunk_end = ftell(f);
    if (chunk_end < 0) {
      status = NM_STATUS_IO_ERROR;
      goto fail;
    }
    uint64_t consumed = (uint64_t)(chunk_end - chunk_start);
    if (consumed > chunk_size) {
      status = NM_STATUS_FORMAT_ERROR;
      goto fail;
    }
    if (consumed < chunk_size && !io_skip_bytes(f, chunk_size - consumed)) {
      status = NM_STATUS_IO_ERROR;
      goto fail;
    }
  }

  if (!has_conf || !has_lays || !has_wght) {
    status = NM_STATUS_FORMAT_ERROR;
    goto fail;
  }

  if (!loaded) {
    build_desc.input_shape = input_shape;
    build_desc.compile_config = compile_cfg;
    build_desc.optimizer_config = opt_cfg;
    build_desc.layout = layout;
    build_desc.layers = descs;
    build_desc.layer_count = (size_t)num_layers_u64;
    loaded = build_model_from_ser_desc(&build_desc);
    if (!loaded) {
      status = NM_STATUS_FORMAT_ERROR;
      goto fail;
    }
  }

  if (tensor_descs != NULL &&
      tensor_desc_count != (uint64_t)nm_seq_tensor_desc_count(loaded)) {
    status = NM_STATUS_FORMAT_ERROR;
    goto fail;
  }

  loaded->training.optimizer_step = loaded_step;
  loaded->runtime.inference_plan.prepared_batch_rows = (size_t)plan_batch_rows;
  loaded->runtime.inference_plan.output_cols = (size_t)plan_output_cols;
  loaded->runtime.inference_plan.peak_tensor_count = (size_t)plan_peak_tensors;
  loaded->runtime.inference_plan.peak_tensor_bytes = (size_t)plan_peak_bytes;
  loaded->runtime.inference_plan.prepared =
      plan_batch_rows > 0 || plan_peak_tensors > 0;
  loaded->storage.format_version = version;
  loaded->storage.alignment = meta_alignment;
  loaded->storage.inference_only =
      (meta_flags & NM_SEQ_FLAG_INFERENCE_ONLY) != 0;
  loaded->storage.loaded_as_bf16 =
      (tensor_descs && tensor_desc_count > 0 &&
       (StDtype)tensor_descs[0].dtype == ST_DTYPE_BF16);
  if (nm_seq_set_autograd_mode(loaded, autograd_mode) != NM_STATUS_OK ||
      nm_seq_set_batch_parallel(loaded, &batch_cfg) != NM_STATUS_OK) {
    status = NM_STATUS_FORMAT_ERROR;
    goto fail;
  }
  loaded->training.inference_only = loaded->storage.inference_only;

  *out_model = loaded;
  loaded = NULL;

fail:
  free(descs);
  free(tensor_descs);
  build_desc.layers = NULL;
  build_desc.tensors = NULL;
  ser_model_desc_destroy(&build_desc);
  nm_seq_free(loaded);

done:
  if (fclose(f) != 0 && status == NM_STATUS_OK)
    status = NM_STATUS_IO_ERROR;
  return status;
}
