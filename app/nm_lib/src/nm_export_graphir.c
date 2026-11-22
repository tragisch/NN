/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm_export_graphir.h"
#include "nm_girw.h"
#include "nm_halfprec.h"
#include "nm_graph_ir_internal.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Writes the model's internal graph as a GraphIR v4 document. Activation
 * values carry a variable batch size (dims[0] == 0) which is replaced by the
 * batch the imported document declared, or else the requested batch size. */

static bool write_shape(FILE *file, const NmIrShape *shape, size_t batch,
                        bool batched) {
  if (fputc('[', file) == EOF)
    return false;
  for (size_t i = 0; i < shape->ndim; ++i) {
    size_t dim = (batched && i == 0) ? batch : shape->dims[i];
    if ((i && fputc(',', file) == EOF) || fprintf(file, "%zu", dim) < 0)
      return false;
  }
  return fputc(']', file) != EOF;
}

static bool write_f32_data(FILE *file, const float *values, size_t count) {
  if (fputs("{\"kind\":\"inline\",\"storage_dtype\":\"float32\",\"values\":[",
            file) == EOF)
    return false;
  for (size_t i = 0; i < count; ++i) {
    if (!isfinite(values[i]) || (i && fputc(',', file) == EOF) ||
        fprintf(file, "%.9g", (double)values[i]) < 0)
      return false;
  }
  return fputs("]}", file) != EOF;
}

/* Where a constant lives in the weight package. */
typedef struct {
  uint64_t offset;
  uint64_t byte_length;
} ExternalRef;

typedef struct {
  const char *package; /* file name the JSON refers to */
  const ExternalRef *refs; /* indexed by value id */
} External;

static bool write_external(FILE *file, const External *external, uint32_t id,
                           const NmIrValue *value) {
  const ExternalRef *ref = &external->refs[id];
  return fprintf(file,
                 "{\"kind\":\"external\",\"package\":\"%s\",\"offset\":%llu,"
                 "\"byte_length\":%llu,\"element_count\":%zu,"
                 "\"storage_dtype\":\"%s\"}",
                 external->package, (unsigned long long)ref->offset,
                 (unsigned long long)ref->byte_length, value->data_count,
                 nm_ir_dtype_name((NmIrDType)value->dtype)) >= 0;
}

/* Inline int64 or bool elements (`raw`). */
static bool write_raw_data(FILE *file, const NmIrValue *value) {
  const bool is_int = value->dtype == NM_IR_DTYPE_I64;
  if (fprintf(file, "{\"kind\":\"inline\",\"storage_dtype\":\"%s\",\"values\":[",
              is_int ? "int64" : "bool") < 0)
    return false;
  for (size_t i = 0; i < value->data_count; ++i) {
    if (i && fputc(',', file) == EOF)
      return false;
    if (is_int ? fprintf(file, "%lld", (long long)((const int64_t *)value->raw)[i]) < 0
               : fputs(((const uint8_t *)value->raw)[i] ? "true" : "false", file) == EOF)
      return false;
  }
  return fputs("]}", file) != EOF;
}

static bool write_value(FILE *file, uint32_t id, const NmIrValue *value,
                        size_t batch, const External *external) {
  const bool batched = value->kind != NM_IR_VALUE_CONSTANT &&
                       value->shape.ndim >= 2 && value->shape.dims[0] == 0;
  if (value->declared_batch)
    batch = value->declared_batch;
  if ((id && fputc(',', file) == EOF) ||
      fprintf(file,
              "{\"id\":%u,\"name\":\"%s\",\"producer\":%u,"
              "\"tensor\":{\"dtype\":\"%s\",\"shape\":",
              id, value->name, value->producer_node_id,
              nm_ir_dtype_name((NmIrDType)value->dtype)) < 0 ||
      !write_shape(file, &value->shape, batch, batched) ||
      fprintf(file,
              ",\"layout\":null,\"quantization\":null,\"name\":\"%s\"},"
              "\"data\":",
              value->name) < 0)
    return false;
  if (value->kind == NM_IR_VALUE_CONSTANT) {
    bool ok = value->raw  ? write_raw_data(file, value)
              : external ? write_external(file, external, id, value)
                         : write_f32_data(file, value->data, value->data_count);
    return ok && fputc('}', file) != EOF;
  }
  return fputs("null}", file) != EOF;
}

static bool write_list(FILE *file, const char *key, const size_t *items,
                       size_t count) {
  if (fprintf(file, "\"%s\":[", key) < 0)
    return false;
  for (size_t i = 0; i < count; ++i) {
    if ((i && fputc(',', file) == EOF) || fprintf(file, "%zu", items[i]) < 0)
      return false;
  }
  return fputc(']', file) != EOF;
}

/* The axes of an axis mask, as a list. */
static size_t mask_axes(uint32_t mask, size_t axes[NM_IR_MAX_DIMS]) {
  size_t count = 0;
  for (size_t i = 0; i < NM_IR_MAX_DIMS; ++i) {
    if (mask & (1u << i))
      axes[count++] = i;
  }
  return count;
}

static bool write_slice(FILE *file, const NmIrNode *node, size_t rank) {
  const char *keys[] = {"starts", "ends", "axes", "steps"};
  for (size_t k = 0; k < 4; ++k) {
    if (fprintf(file, "%s\"%s\":[", k ? "," : "", keys[k]) < 0)
      return false;
    bool first = true;
    for (size_t i = 0; i < rank; ++i) {
      if (node->cfg.slice.starts[i] == 0 && node->cfg.slice.ends[i] == INT64_MAX &&
          node->cfg.slice.steps[i] == 1)
        continue; /* axis not sliced */
      long long item = k == 0   ? (long long)node->cfg.slice.starts[i]
                       : k == 1 ? (long long)node->cfg.slice.ends[i]
                       : k == 2 ? (long long)i
                                : (long long)node->cfg.slice.steps[i];
      if ((!first && fputc(',', file) == EOF) || fprintf(file, "%lld", item) < 0)
        return false;
      first = false;
    }
    if (fputc(']', file) == EOF)
      return false;
  }
  return true;
}

static bool write_attributes(FILE *file, const NmIrGraph *graph,
                             const NmIrNode *node, size_t batch) {
  size_t axes[NM_IR_MAX_DIMS];
  const NmIrValue *output = &graph->values[node->output_id];
  switch (node->op_type) {
  case NM_IR_OP_GATHER:
    return fprintf(file, "\"axis\":%zu", node->cfg.gather.axis) >= 0;
  case NM_IR_OP_CAST:
    return fprintf(file, "\"to\":\"%s\"",
                   nm_ir_dtype_name((NmIrDType)output->dtype)) >= 0;
  case NM_IR_OP_EXPAND:
    return write_list(file, "shape", node->cfg.expand.dims, node->cfg.expand.ndim);
  case NM_IR_OP_SLICE:
    return write_slice(file, node, graph->values[node->input_ids[0]].shape.ndim);
  case NM_IR_OP_REDUCEMAX:
    return write_list(file, "axes", axes, mask_axes(node->cfg.axes.axes, axes)) &&
           fprintf(file, ",\"keepdims\":%s",
                   node->cfg.axes.keepdims ? "true" : "false") >= 0;
  case NM_IR_OP_SQUEEZE:
  case NM_IR_OP_UNSQUEEZE:
    return write_list(file, "axes", axes, mask_axes(node->cfg.axes.axes, axes));
  case NM_IR_OP_RANGE: {
    /* Any limit that gives the same length is equivalent. */
    const double limit = node->cfg.range.start +
                         (double)output->shape.dims[0] * node->cfg.range.delta;
    return fprintf(file, "\"start\":%.17g,\"limit\":%.17g,\"delta\":%.17g",
                   node->cfg.range.start, limit, node->cfg.range.delta) >= 0;
  }
  case NM_IR_OP_CONV2D:
    return fprintf(file,
                   "\"strides\":[%zu,%zu],\"pads\":[%zu,%zu,%zu,%zu],"
                   "\"dilations\":[1,1],\"groups\":1",
                   node->cfg.conv2d.stride_h, node->cfg.conv2d.stride_w,
                   node->cfg.conv2d.pad_h, node->cfg.conv2d.pad_w,
                   node->cfg.conv2d.pad_h, node->cfg.conv2d.pad_w) >= 0;
  case NM_IR_OP_MAXPOOL2D:
    return fprintf(file,
                   "\"kernel\":[%zu,%zu],\"strides\":[%zu,%zu],"
                   "\"pads\":[%zu,%zu,%zu,%zu]",
                   node->cfg.maxpool2d.kernel_h, node->cfg.maxpool2d.kernel_w,
                   node->cfg.maxpool2d.stride_h, node->cfg.maxpool2d.stride_w,
                   node->cfg.maxpool2d.pad_h, node->cfg.maxpool2d.pad_w,
                   node->cfg.maxpool2d.pad_h, node->cfg.maxpool2d.pad_w) >= 0;
  case NM_IR_OP_RESHAPE:
    return fputs("\"shape\":", file) != EOF &&
           write_shape(file, &output->shape,
                       output->declared_batch ? output->declared_batch : batch,
                       output->shape.ndim > 0 && output->shape.dims[0] == 0);
  case NM_IR_OP_TRANSPOSE: {
    if (fputs("\"perm\":[", file) == EOF)
      return false;
    size_t rank = graph->values[node->output_id].shape.ndim;
    for (size_t i = 0; i < rank; ++i) {
      if ((i && fputc(',', file) == EOF) ||
          fprintf(file, "%zu", node->cfg.transpose.perm[i]) < 0)
        return false;
    }
    return fputc(']', file) != EOF;
  }
  case NM_IR_OP_LAYERNORM: {
    const NmIrShape *input = &graph->values[node->input_ids[0]].shape;
    if (fputs("\"normalized_shape\":[", file) == EOF)
      return false;
    for (size_t i = input->ndim - node->cfg.layernorm.norm_rank; i < input->ndim;
         ++i) {
      if ((i > input->ndim - node->cfg.layernorm.norm_rank &&
           fputc(',', file) == EOF) ||
          fprintf(file, "%zu", input->dims[i]) < 0)
        return false;
    }
    return fprintf(file, "],\"epsilon\":%.9g",
                   (double)node->cfg.layernorm.epsilon) >= 0;
  }
  case NM_IR_OP_GELU:
    return fprintf(file, "\"approximate\":\"%s\"",
                   node->cfg.gelu.exact ? "none" : "tanh") >= 0;
  default:
    return true;
  }
}

static bool write_node(FILE *file, const NmIrGraph *graph, uint32_t id,
                       size_t batch) {
  const NmIrNode *node = &graph->nodes[id];
  const NmIrValue *output = &graph->values[node->output_id];
  if ((id && fputc(',', file) == EOF) ||
      fprintf(file,
              "{\"id\":%u,\"name\":\"%s\",\"operation\":\"%s\","
              "\"inputs\":[",
              id, output->name, nm_ir_op_name((NmIrOpType)node->op_type)) < 0)
    return false;
  for (uint32_t i = 0; i < node->input_count; ++i) {
    if ((i && fputc(',', file) == EOF) ||
        fprintf(file, "%u", node->input_ids[i]) < 0)
      return false;
  }
  return fprintf(file, "],\"outputs\":[%u],\"attributes\":{", node->output_id) >=
             0 &&
         write_attributes(file, graph, node, batch) &&
         fputs("},\"metadata\":{}}", file) != EOF;
}

static NmStatus validate_exportable(const NmIrGraph *graph,
                                    const NmGraphIrExportOptions *options) {
  if (!graph || !options || options->batch_size == 0)
    return NM_STATUS_INVALID_ARGUMENT;
  if (graph->node_count == 0 || graph->model_input_id >= graph->value_count ||
      graph->model_output_id >= graph->value_count)
    return NM_STATUS_INVALID_ARGUMENT;

  for (size_t i = 0; i < graph->value_count; ++i) {
    const NmIrValue *value = &graph->values[i];
    /* Inline JSON data is float32, int64 or bool; 16-bit types need a
     * weight package. */
    if ((value->dtype == NM_IR_DTYPE_BF16 || value->dtype == NM_IR_DTYPE_F16) &&
        !options->weights_path)
      return NM_STATUS_INVALID_ARGUMENT;
    if (value->kind != NM_IR_VALUE_CONSTANT || value->raw)
      continue;
    if (!value->data)
      return NM_STATUS_UNSUPPORTED_TYPE;
    for (size_t j = 0; j < value->data_count; ++j) {
      if (!isfinite(value->data[j]))
        return NM_STATUS_FORMAT_ERROR;
    }
  }
  return NM_STATUS_OK;
}

/* The lowest opset that has every operation of the graph. */
static int needed_opset(const NmIrGraph *graph) {
  int opset = 1;
  for (size_t i = 0; i < graph->node_count; ++i) {
    switch (graph->nodes[i].op_type) {
    case NM_IR_OP_EQUAL:
    case NM_IR_OP_NOT:
    case NM_IR_OP_CAST:
    case NM_IR_OP_RANGE:
      return 3;
    case NM_IR_OP_GATHER:
    case NM_IR_OP_WHERE:
    case NM_IR_OP_EXPAND:
    case NM_IR_OP_SLICE:
    case NM_IR_OP_REDUCEMAX:
    case NM_IR_OP_SQUEEZE:
    case NM_IR_OP_UNSQUEEZE:
    case NM_IR_OP_CONV2D:
    case NM_IR_OP_MAXPOOL2D:
    case NM_IR_OP_SUB:
    case NM_IR_OP_MUL:
    case NM_IR_OP_DIV:
    case NM_IR_OP_LAYERNORM:
    case NM_IR_OP_GELU:
      opset = 2;
      break;
    case NM_IR_OP_MATMUL: /* batched MatMul is opset 2 */
      for (uint32_t j = 0; j < graph->nodes[i].input_count; ++j) {
        if (graph->values[graph->nodes[i].input_ids[j]].shape.ndim > 2)
          opset = 2;
      }
      break;
    default:
      break;
    }
  }
  return opset;
}

/* Writes all constants into a weight package and records where each one is. */
static NmStatus write_weight_package(const NmIrGraph *graph, const char *path,
                                     ExternalRef **out_refs) {
  size_t constants = 0;
  for (size_t i = 0; i < graph->value_count; ++i)
    constants += graph->values[i].kind == NM_IR_VALUE_CONSTANT;
  NmGirwSource *sources = calloc(constants ? constants : 1, sizeof(*sources));
  ExternalRef *refs = calloc(graph->value_count, sizeof(*refs));
  size_t *ids = calloc(constants ? constants : 1, sizeof(*ids));
  uint16_t **narrow = calloc(constants ? constants : 1, sizeof(*narrow));
  void *image = NULL;
  NmGirwPackage *package = NULL;
  NmStatus status = NM_STATUS_ALLOC_FAILED;
  if (!sources || !refs || !ids || !narrow)
    goto done;

  size_t n = 0;
  for (size_t i = 0; i < graph->value_count; ++i) {
    const NmIrValue *v = &graph->values[i];
    if (v->kind != NM_IR_VALUE_CONSTANT || v->raw)
      continue; /* int64 and bool constants stay inline */
    sources[n] = (NmGirwSource){.name = v->name, .dtype = NM_GIRW_FLOAT32,
                                .element_count = v->data_count, .data = v->data};
    if (v->dtype != NM_IR_DTYPE_F32) {
      /* Weights of a 16-bit model go back to their own type (rounded). */
      uint16_t *bits = malloc((v->data_count ? v->data_count : 1) * sizeof(*bits));
      if (!bits)
        goto done;
      for (size_t k = 0; k < v->data_count; ++k)
        bits[k] = v->dtype == NM_IR_DTYPE_BF16 ? nm_f32_to_bf16(v->data[k])
                                               : nm_f32_to_f16(v->data[k]);
      narrow[n] = bits;
      sources[n].dtype = v->dtype == NM_IR_DTYPE_BF16 ? NM_GIRW_BFLOAT16
                                                      : NM_GIRW_FLOAT16;
      sources[n].data = bits;
    }
    ids[n++] = i;
  }
  size_t size = nm_girw_image_size(sources, n);
  image = size ? malloc(size) : NULL;
  if (!image) {
    status = size ? NM_STATUS_ALLOC_FAILED : NM_STATUS_FORMAT_ERROR;
    goto done;
  }
  status = nm_girw_write_image(sources, n, image, size);
  if (status != NM_STATUS_OK)
    goto done;
  status = nm_girw_open_memory(image, size, NM_GIRW_VERIFY_INDEX_ONLY, &package);
  if (status != NM_STATUS_OK)
    goto done;
  for (size_t e = 0; e < nm_girw_entry_count(package); ++e) {
    const NmGirwEntry *entry = nm_girw_entry(package, e);
    for (size_t k = 0; k < n; ++k) {
      if (!strcmp(entry->name, sources[k].name)) {
        refs[ids[k]] = (ExternalRef){entry->offset, entry->byte_length};
        break;
      }
    }
  }
  FILE *file = fopen(path, "wb");
  if (!file) {
    status = NM_STATUS_IO_ERROR;
    goto done;
  }
  bool ok = fwrite(image, 1, size, file) == size;
  if (fclose(file) != 0)
    ok = false;
  status = ok ? NM_STATUS_OK : NM_STATUS_IO_ERROR;
  if (ok) {
    *out_refs = refs;
    refs = NULL;
  }
done:
  nm_girw_close(package);
  if (narrow) {
    for (size_t k = 0; k < constants; ++k)
      free(narrow[k]);
  }
  free(narrow);
  free(image);
  free(sources);
  free(ids);
  free(refs);
  return status;
}

NmStatus nm_export_graphir_graph_json(const NmGraphModel *graph,
                                      const NmGraphIrExportOptions *options,
                                      const char *path) {
  if (!path)
    return NM_STATUS_INVALID_ARGUMENT;
  NmStatus status = validate_exportable(graph, options);
  if (status != NM_STATUS_OK)
    return status;

  /* The weights are written first so a failure leaves no half-written JSON
   * behind. Entries are named after the (unique) value names. */
  External external = {0};
  const External *external_info = NULL;
  ExternalRef *refs = NULL;
  if (options->weights_path) {
    status = write_weight_package(graph, options->weights_path, &refs);
    if (status != NM_STATUS_OK)
      return status;
    const char *slash = strrchr(options->weights_path, '/');
    external = (External){.package = slash ? slash + 1 : options->weights_path,
                          .refs = refs};
    external_info = &external;
  }

  FILE *file = fopen(path, "wb");
  if (!file) {
    free(refs);
    return NM_STATUS_IO_ERROR;
  }

  bool ok = fprintf(file,
                    "{\"format_version\":4,\"opset_version\":%d,"
                    "\"name\":\"nm.frontend\",\"metadata\":{},"
                    "\"inputs\":[",
                    needed_opset(graph)) >= 0;
  const size_t inputs = graph->input_count ? graph->input_count : 1;
  for (size_t i = 0; ok && i < inputs; ++i)
    ok = fprintf(file, "%s%u", i ? "," : "",
                 graph->input_count ? graph->input_ids[i] : graph->model_input_id) >= 0;
  ok = ok && fprintf(file, "],\"outputs\":[%u],\"values\":[",
                     graph->model_output_id) >= 0;
  for (size_t i = 0; ok && i < graph->value_count; ++i)
    ok = write_value(file, (uint32_t)i, &graph->values[i], options->batch_size,
                     external_info);
  ok = ok && fputs("],\"nodes\":[", file) != EOF;
  for (size_t i = 0; ok && i < graph->node_count; ++i)
    ok = write_node(file, graph, (uint32_t)i, options->batch_size);
  ok = ok && fputs("]}", file) != EOF;
  if (fclose(file) != 0)
    ok = false;
  free(refs);
  return ok ? NM_STATUS_OK : NM_STATUS_IO_ERROR;
}

NmStatus nm_export_graphir_json(const NmFrontendModel *model,
                                const NmGraphIrExportOptions *options,
                                const char *path) {
  const NmGraphModel *graph =
      nm_frontend_graph((NmFrontendModel *)(uintptr_t)model);
  return nm_export_graphir_graph_json(graph, options, path);
}
