/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm_import_graphir.h"
#include "nm_girw.h"
#include "nm_halfprec.h"
#include "nm_graph_ir_internal.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Minimal JSON tree --------------------------------------------------- */

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } JType;

typedef struct Json {
  JType type;
  double num;
  char *str;
  struct Json *items;
  char **keys;
  size_t count;
  size_t capacity;
} Json;

typedef struct {
  const char *cur;
  const char *end;
  bool failed;
} Parser;

static void json_free_children(Json *node) {
  for (size_t i = 0; i < node->count; ++i) {
    json_free_children(&node->items[i]);
    if (node->keys)
      free(node->keys[i]);
  }
  free(node->items);
  free(node->keys);
  free(node->str);
}

static void skip_ws(Parser *p) {
  while (p->cur < p->end && isspace((unsigned char)*p->cur))
    p->cur++;
}

static bool json_push(Json *node, Json child, char *key) {
  if (node->count == node->capacity) {
    size_t capacity = node->capacity ? node->capacity * 2 : 8;
    Json *items = realloc(node->items, capacity * sizeof(*items));
    if (!items)
      return false;
    node->items = items;
    if (node->type == J_OBJ) {
      char **keys = realloc(node->keys, capacity * sizeof(*keys));
      if (!keys)
        return false;
      node->keys = keys;
    }
    node->capacity = capacity;
  }
  node->items[node->count] = child;
  if (node->type == J_OBJ)
    node->keys[node->count] = key;
  node->count++;
  return true;
}

static char *parse_string(Parser *p) {
  if (p->cur >= p->end || *p->cur != '"') {
    p->failed = true;
    return NULL;
  }
  p->cur++;
  const char *start = p->cur;
  while (p->cur < p->end && *p->cur != '"') {
    if (*p->cur == '\\')
      p->cur++;
    p->cur++;
  }
  if (p->cur >= p->end) {
    p->failed = true;
    return NULL;
  }
  size_t length = (size_t)(p->cur - start);
  char *text = malloc(length + 1);
  if (!text) {
    p->failed = true;
    return NULL;
  }
  size_t out = 0;
  for (size_t i = 0; i < length; ++i) {
    if (start[i] == '\\' && i + 1 < length)
      i++;
    text[out++] = start[i];
  }
  text[out] = '\0';
  p->cur++;
  return text;
}

static Json parse_value(Parser *p, int depth) {
  Json node = {0};
  skip_ws(p);
  if (p->failed || depth > 32 || p->cur >= p->end) {
    p->failed = true;
    return node;
  }
  char c = *p->cur;
  if (c == '{' || c == '[') {
    bool object = c == '{';
    node.type = object ? J_OBJ : J_ARR;
    char close = object ? '}' : ']';
    p->cur++;
    skip_ws(p);
    if (p->cur < p->end && *p->cur == close) {
      p->cur++;
      return node;
    }
    while (!p->failed) {
      char *key = NULL;
      if (object) {
        skip_ws(p);
        key = parse_string(p);
        skip_ws(p);
        if (p->failed || p->cur >= p->end || *p->cur != ':') {
          free(key);
          p->failed = true;
          break;
        }
        p->cur++;
      }
      Json child = parse_value(p, depth + 1);
      if (p->failed || !json_push(&node, child, key)) {
        json_free_children(&child);
        free(key);
        p->failed = true;
        break;
      }
      skip_ws(p);
      if (p->cur < p->end && *p->cur == ',') {
        p->cur++;
        continue;
      }
      if (p->cur < p->end && *p->cur == close) {
        p->cur++;
        return node;
      }
      p->failed = true;
    }
    return node;
  }
  if (c == '"') {
    node.type = J_STR;
    node.str = parse_string(p);
    return node;
  }
  if (!strncmp(p->cur, "null", 4) && p->end - p->cur >= 4) {
    p->cur += 4;
    return node;
  }
  if (!strncmp(p->cur, "true", 4) && p->end - p->cur >= 4) {
    node.type = J_BOOL;
    node.num = 1;
    p->cur += 4;
    return node;
  }
  if (!strncmp(p->cur, "false", 5) && p->end - p->cur >= 5) {
    node.type = J_BOOL;
    p->cur += 5;
    return node;
  }
  char *end = NULL;
  node.num = strtod(p->cur, &end);
  if (end == p->cur) {
    p->failed = true;
    return node;
  }
  node.type = J_NUM;
  p->cur = end;
  return node;
}

static const Json *json_get(const Json *object, const char *key) {
  if (!object || object->type != J_OBJ)
    return NULL;
  for (size_t i = 0; i < object->count; ++i) {
    if (!strcmp(object->keys[i], key))
      return &object->items[i];
  }
  return NULL;
}

static bool json_ints(const Json *array, size_t *out, size_t max,
                      size_t *count) {
  if (!array || array->type != J_ARR || array->count > max)
    return false;
  for (size_t i = 0; i < array->count; ++i) {
    const Json *item = &array->items[i];
    if (item->type != J_NUM || item->num < 0 || item->num != (double)(size_t)item->num)
      return false;
    out[i] = (size_t)item->num;
  }
  *count = array->count;
  return true;
}

/* Signed integers (axes, slice bounds). */
static bool json_i64s(const Json *array, int64_t *out, size_t max,
                      size_t *count) {
  if (!array || array->type != J_ARR || array->count > max)
    return false;
  for (size_t i = 0; i < array->count; ++i) {
    const Json *item = &array->items[i];
    if (item->type != J_NUM || item->num != (double)(int64_t)item->num)
      return false;
    out[i] = (int64_t)item->num;
  }
  *count = array->count;
  return true;
}

/* ---- GraphIR interpretation --------------------------------------------- */

#define MAX_DIMS NM_IR_MAX_DIMS

typedef struct {
  size_t dims[MAX_DIMS];
  size_t ndim;
  const Json *data;     /* array of numbers, or NULL */
  const Json *external; /* external data descriptor object, or NULL */
  int dtype;            /* NmIrDType, or -1 if nm_lib cannot run it */
} ValueInfo;

typedef struct {
  const Json *node;
  const char *op;
  size_t inputs[3];
  size_t input_count;
  size_t output;
  const Json *attrs;
} NodeInfo;

typedef struct {
  ValueInfo *values;
  size_t value_count;
  NodeInfo *nodes;
  size_t node_count;
} Graph;

/* Inline values whose storage type is `dtype` ("float32", "int64", "bool"). */
static const Json *inline_values(const Json *data, const char *dtype_name) {
  if (!data)
    return NULL;
  if (data->type == J_ARR)
    return !strcmp(dtype_name, "float32") ? data : NULL; /* format v1-v3 */
  const Json *kind = json_get(data, "kind");
  const Json *dtype = json_get(data, "storage_dtype");
  const Json *values = json_get(data, "values");
  if (kind && kind->type == J_STR && !strcmp(kind->str, "inline") && dtype &&
      dtype->type == J_STR && !strcmp(dtype->str, dtype_name) && values &&
      values->type == J_ARR)
    return values;
  return NULL;
}

static bool load_graph(const Json *root, Graph *graph) {
  const Json *values = json_get(root, "values");
  const Json *nodes = json_get(root, "nodes");
  if (!values || values->type != J_ARR || !nodes || nodes->type != J_ARR)
    return false;
  graph->values = calloc(values->count ? values->count : 1,
                         sizeof(*graph->values));
  graph->nodes = calloc(nodes->count ? nodes->count : 1,
                        sizeof(*graph->nodes));
  if (!graph->values || !graph->nodes)
    return false;
  graph->value_count = values->count;
  graph->node_count = nodes->count;
  for (size_t i = 0; i < values->count; ++i) {
    const Json *tensor = json_get(&values->items[i], "tensor");
    const Json *dtype = json_get(tensor, "dtype");
    if (!dtype || dtype->type != J_STR ||
        !json_ints(json_get(tensor, "shape"), graph->values[i].dims, MAX_DIMS,
                   &graph->values[i].ndim))
      return false;
    /* Other types are well-formed but not runnable; reported as unsupported
     * once the whole document has been read. */
    graph->values[i].dtype = !strcmp(dtype->str, "float32")    ? NM_IR_DTYPE_F32
                             : !strcmp(dtype->str, "bfloat16") ? NM_IR_DTYPE_BF16
                             : !strcmp(dtype->str, "float16")  ? NM_IR_DTYPE_F16
                             : !strcmp(dtype->str, "int64")    ? NM_IR_DTYPE_I64
                             : !strcmp(dtype->str, "bool")     ? NM_IR_DTYPE_BOOL
                                                               : -1;
    const Json *data = json_get(&values->items[i], "data");
    const Json *kind = json_get(data, "kind");
    if (kind && kind->type == J_STR && !strcmp(kind->str, "external"))
      graph->values[i].external = data;
    else
      graph->values[i].data = inline_values(
          data, graph->values[i].dtype == NM_IR_DTYPE_I64    ? "int64"
                : graph->values[i].dtype == NM_IR_DTYPE_BOOL ? "bool"
                                                             : "float32");
  }
  for (size_t i = 0; i < nodes->count; ++i) {
    NodeInfo *info = &graph->nodes[i];
    const Json *op = json_get(&nodes->items[i], "operation");
    const Json *outputs = json_get(&nodes->items[i], "outputs");
    size_t output_count = 0;
    size_t output[1];
    if (!op || op->type != J_STR ||
        !json_ints(json_get(&nodes->items[i], "inputs"), info->inputs, 3,
                   &info->input_count) ||
        !json_ints(outputs, output, 1, &output_count) || output_count != 1 ||
        output[0] >= graph->value_count)
      return false;
    for (size_t j = 0; j < info->input_count; ++j) {
      if (info->inputs[j] >= graph->value_count)
        return false;
    }
    info->node = &nodes->items[i];
    info->op = op->str;
    info->output = output[0];
    info->attrs = json_get(&nodes->items[i], "attributes");
  }
  return true;
}

static bool int_attr(const Json *attrs, const char *key, size_t *out,
                     size_t max, size_t *count) {
  return json_ints(json_get(attrs, key), out, max, count);
}

static size_t shape_count(const NmIrShape *shape) {
  size_t count = 1;
  for (size_t i = 0; i < shape->ndim; ++i)
    count *= shape->dims[i];
  return count;
}

/* Weight packages opened during one import, shared between constants. */
typedef struct {
  const char *directory; /* directory of the GraphIR document */
  struct {
    char *path;
    NmGirwPackage *package;
  } *entries;
  size_t count;
} PackageCache;

static void package_cache_free(PackageCache *cache) {
  for (size_t i = 0; i < cache->count; ++i) {
    nm_girw_close(cache->entries[i].package);
    free(cache->entries[i].path);
  }
  free(cache->entries);
}

static NmStatus package_cache_get(PackageCache *cache, const char *name,
                                  NmGirwPackage **out) {
  size_t length = strlen(name) + strlen(cache->directory) + 2;
  char *path = malloc(length);
  if (!path)
    return NM_STATUS_ALLOC_FAILED;
  if (name[0] == '/')
    snprintf(path, length, "%s", name);
  else
    snprintf(path, length, "%s/%s", cache->directory, name);
  for (size_t i = 0; i < cache->count; ++i) {
    if (!strcmp(cache->entries[i].path, path)) {
      free(path);
      *out = cache->entries[i].package;
      return NM_STATUS_OK;
    }
  }
  NmGirwPackage *package = NULL;
  NmStatus status = nm_girw_open(path, &package);
  if (status != NM_STATUS_OK) {
    free(path);
    return status;
  }
  void *grown = realloc(cache->entries, (cache->count + 1) * sizeof(*cache->entries));
  if (!grown) {
    nm_girw_close(package);
    free(path);
    return NM_STATUS_ALLOC_FAILED;
  }
  cache->entries = grown;
  cache->entries[cache->count].path = path;
  cache->entries[cache->count].package = package;
  cache->count++;
  *out = package;
  return NM_STATUS_OK;
}

static bool host_is_little_endian(void) {
  const uint16_t probe = 1;
  return *(const uint8_t *)&probe == 1;
}

/* Allocates graph-owned storage for constant `value` (a tensor for rank 1
 * and 4, plain data otherwise) and returns the float destination. */
static NmStatus allocate_constant(NmIrValue *value, size_t count, float **dest) {
  if (count == 0)
    return NM_STATUS_FORMAT_ERROR;
  if (value->shape.ndim == 1 || value->shape.ndim == 4) {
    value->tensor = st_create(value->shape.ndim, value->shape.dims);
    if (!value->tensor)
      return NM_STATUS_ALLOC_FAILED;
    *dest = value->tensor->values;
    value->data = *dest;
  } else {
    *dest = malloc(count * sizeof(float));
    if (!*dest)
      return NM_STATUS_ALLOC_FAILED;
    value->data = *dest;
  }
  value->owns_data = true;
  value->data_count = count;
  return NM_STATUS_OK;
}

static bool external_u64(const Json *descriptor, const char *key, uint64_t *out) {
  const Json *field = json_get(descriptor, key);
  if (!field || field->type != J_NUM || field->num < 0 ||
      field->num >= 18446744073709551616.0 || field->num != (double)(uint64_t)field->num)
    return false;
  *out = (uint64_t)field->num;
  return true;
}

/* Copies weights from a `.girw` package into graph-owned f32 storage. The
 * storage type must be the value's own type; 16-bit weights are widened. */
static NmStatus attach_external(NmIrValue *value, const Json *descriptor,
                                PackageCache *cache) {
  const Json *package_name = json_get(descriptor, "package");
  const Json *dtype = json_get(descriptor, "storage_dtype");
  uint64_t offset, byte_length, elements;
  if (!package_name || package_name->type != J_STR || !dtype ||
      dtype->type != J_STR || !external_u64(descriptor, "offset", &offset) ||
      !external_u64(descriptor, "byte_length", &byte_length) ||
      !external_u64(descriptor, "element_count", &elements))
    return NM_STATUS_FORMAT_ERROR;
  if (!host_is_little_endian())
    return NM_STATUS_UNSUPPORTED_TYPE;
  NmGirwDType stored;
  if (!strcmp(dtype->str, "float32"))
    stored = NM_GIRW_FLOAT32;
  else if (!strcmp(dtype->str, "bfloat16"))
    stored = NM_GIRW_BFLOAT16;
  else if (!strcmp(dtype->str, "float16"))
    stored = NM_GIRW_FLOAT16;
  else
    return NM_STATUS_UNSUPPORTED_TYPE;
  const NmGirwDType expected = value->dtype == NM_IR_DTYPE_BF16  ? NM_GIRW_BFLOAT16
                               : value->dtype == NM_IR_DTYPE_F16 ? NM_GIRW_FLOAT16
                                                                 : NM_GIRW_FLOAT32;
  if (stored != expected)
    return NM_STATUS_FORMAT_ERROR; /* storage must match the tensor type */
  size_t count = shape_count(&value->shape);
  if (elements != count)
    return NM_STATUS_FORMAT_ERROR;

  NmGirwPackage *package = NULL;
  NmStatus status = package_cache_get(cache, package_name->str, &package);
  if (status != NM_STATUS_OK)
    return status;
  const NmGirwEntry *entry =
      nm_girw_find(package, offset, byte_length, elements, stored);
  if (!entry)
    return NM_STATUS_FORMAT_ERROR;
  float *dest = NULL;
  status = allocate_constant(value, count, &dest);
  if (status != NM_STATUS_OK)
    return status;
  if (stored == NM_GIRW_FLOAT32) {
    memcpy(dest, entry->data, count * sizeof(float));
  } else {
    for (size_t i = 0; i < count; ++i) {
      uint16_t bits;
      memcpy(&bits, entry->data + 2 * i, sizeof(bits));
      dest[i] = stored == NM_GIRW_BFLOAT16 ? nm_bf16_to_f32(bits)
                                           : nm_f16_to_f32(bits);
    }
  }
  return NM_STATUS_OK;
}

/* Copies inline int64 or bool values into `value->raw`. */
static NmStatus attach_raw_constant(NmIrValue *value, const ValueInfo *info) {
  size_t count = shape_count(&value->shape);
  if (info->external || !info->data || info->data->count != count)
    return NM_STATUS_FORMAT_ERROR; /* packages hold float weights only */
  bool is_int = value->dtype == NM_IR_DTYPE_I64;
  value->raw = malloc(count * (is_int ? sizeof(int64_t) : 1));
  if (!value->raw)
    return NM_STATUS_ALLOC_FAILED;
  value->data_count = count;
  for (size_t i = 0; i < count; ++i) {
    const Json *item = &info->data->items[i];
    if (is_int) {
      if (item->type != J_NUM || item->num != (double)(int64_t)item->num)
        return NM_STATUS_FORMAT_ERROR;
      ((int64_t *)value->raw)[i] = (int64_t)item->num;
    } else {
      if (item->type != J_BOOL)
        return NM_STATUS_FORMAT_ERROR;
      ((uint8_t *)value->raw)[i] = item->num != 0;
    }
  }
  return NM_STATUS_OK;
}

/* Copies a numeric JSON array into graph-owned storage for constant `value`. */
static NmStatus attach_constant(NmIrValue *value, const ValueInfo *info,
                                PackageCache *cache) {
  if (value->dtype == NM_IR_DTYPE_I64 || value->dtype == NM_IR_DTYPE_BOOL)
    return attach_raw_constant(value, info);
  if (info->external)
    return attach_external(value, info->external, cache);
  if (value->dtype != NM_IR_DTYPE_F32)
    return NM_STATUS_FORMAT_ERROR; /* inline data is float32 only */
  size_t count = shape_count(&value->shape);
  if (!info->data || info->data->count != count)
    return NM_STATUS_FORMAT_ERROR;
  float *dest = NULL;
  NmStatus status = allocate_constant(value, count, &dest);
  if (status != NM_STATUS_OK)
    return status;
  for (size_t i = 0; i < count; ++i) {
    if (info->data->items[i].type != J_NUM)
      return NM_STATUS_FORMAT_ERROR;
    dest[i] = (float)info->data->items[i].num;
  }
  return NM_STATUS_OK;
}

/* Normalizes a possibly negative `axis` against `rank`. */
static bool normalize_axis(int64_t axis, size_t rank, size_t *out) {
  int64_t r = (int64_t)rank;
  if (axis < -r || axis >= r)
    return false;
  *out = (size_t)(axis < 0 ? axis + r : axis);
  return true;
}

/* Reads the `axes` list as a bit mask over `rank` axes; duplicates fail. */
static bool axes_mask(const Json *attrs, size_t rank, uint32_t *mask) {
  int64_t axes[MAX_DIMS];
  size_t n = 0, axis = 0;
  if (!json_i64s(json_get(attrs, "axes"), axes, MAX_DIMS, &n) || n == 0)
    return false;
  *mask = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!normalize_axis(axes[i], rank, &axis) || (*mask & (1u << axis)))
      return false;
    *mask |= 1u << axis;
  }
  return true;
}

static int dtype_from_name(const char *name) {
  if (!strcmp(name, "float32") || !strcmp(name, "bfloat16") ||
      !strcmp(name, "float16"))
    return NM_IR_DTYPE_F32; /* float types compute in f32 */
  if (!strcmp(name, "int64"))
    return NM_IR_DTYPE_I64;
  if (!strcmp(name, "bool"))
    return NM_IR_DTYPE_BOOL;
  return -1;
}

static bool number_attr(const Json *attrs, const char *key, double *out) {
  const Json *value = json_get(attrs, key);
  if (!value || value->type != J_NUM)
    return false;
  *out = value->num;
  return true;
}

/* Configuration of the tensor operations (Gather ... Unsqueeze). */
static NmStatus read_tensor_op_config(const NodeInfo *info, NmIrNode *node,
                                      const Graph *source) {
  const size_t rank =
      info->input_count ? source->values[info->inputs[0]].ndim : 0;
  const ValueInfo *output = &source->values[info->output];
  switch (node->op_type) {
  case NM_IR_OP_GATHER: {
    const Json *axis = json_get(info->attrs, "axis");
    if (!axis || axis->type != J_NUM || axis->num != (double)(int64_t)axis->num)
      return NM_STATUS_FORMAT_ERROR;
    return normalize_axis((int64_t)axis->num, rank, &node->cfg.gather.axis)
               ? NM_STATUS_OK
               : NM_STATUS_FORMAT_ERROR;
  }
  case NM_IR_OP_CAST: {
    const Json *to = json_get(info->attrs, "to");
    if (!to || to->type != J_STR)
      return NM_STATUS_FORMAT_ERROR;
    int dtype = dtype_from_name(to->str);
    if (dtype < 0)
      return NM_STATUS_UNSUPPORTED_TYPE; /* int32 and friends */
    node->cfg.cast.to = (uint8_t)dtype;
    return NM_STATUS_OK;
  }
  case NM_IR_OP_EXPAND:
    return int_attr(info->attrs, "shape", node->cfg.expand.dims, MAX_DIMS,
                    &node->cfg.expand.ndim)
               ? NM_STATUS_OK
               : NM_STATUS_UNSUPPORTED_TYPE;
  case NM_IR_OP_SLICE: {
    int64_t starts[MAX_DIMS], ends[MAX_DIMS], axes[MAX_DIMS], steps[MAX_DIMS];
    size_t n = 0, n_ends = 0, n_axes = 0, n_steps = 0;
    if (!json_i64s(json_get(info->attrs, "starts"), starts, MAX_DIMS, &n) ||
        !json_i64s(json_get(info->attrs, "ends"), ends, MAX_DIMS, &n_ends) ||
        n_ends != n)
      return NM_STATUS_FORMAT_ERROR;
    if (json_get(info->attrs, "axes")) {
      if (!json_i64s(json_get(info->attrs, "axes"), axes, MAX_DIMS, &n_axes) ||
          n_axes != n)
        return NM_STATUS_FORMAT_ERROR;
    } else {
      for (size_t i = 0; i < n; ++i)
        axes[i] = (int64_t)i;
    }
    if (json_get(info->attrs, "steps")) {
      if (!json_i64s(json_get(info->attrs, "steps"), steps, MAX_DIMS, &n_steps) ||
          n_steps != n)
        return NM_STATUS_FORMAT_ERROR;
    } else {
      for (size_t i = 0; i < n; ++i)
        steps[i] = 1;
    }
    for (size_t i = 0; i < MAX_DIMS; ++i) {
      node->cfg.slice.starts[i] = 0;
      node->cfg.slice.ends[i] = INT64_MAX;
      node->cfg.slice.steps[i] = 1;
    }
    uint32_t seen = 0;
    for (size_t i = 0; i < n; ++i) {
      size_t axis = 0;
      if (!normalize_axis(axes[i], rank, &axis) || (seen & (1u << axis)) ||
          steps[i] == 0)
        return NM_STATUS_FORMAT_ERROR;
      seen |= 1u << axis;
      node->cfg.slice.starts[axis] = starts[i];
      node->cfg.slice.ends[axis] = ends[i];
      node->cfg.slice.steps[axis] = steps[i];
    }
    return NM_STATUS_OK;
  }
  case NM_IR_OP_REDUCEMAX: {
    const Json *keepdims = json_get(info->attrs, "keepdims");
    if (keepdims && keepdims->type != J_BOOL && keepdims->type != J_NUM)
      return NM_STATUS_FORMAT_ERROR;
    node->cfg.axes.keepdims = !keepdims || keepdims->num != 0;
    return axes_mask(info->attrs, rank, &node->cfg.axes.axes)
               ? NM_STATUS_OK
               : NM_STATUS_FORMAT_ERROR;
  }
  case NM_IR_OP_SQUEEZE:
    return axes_mask(info->attrs, rank, &node->cfg.axes.axes)
               ? NM_STATUS_OK
               : NM_STATUS_FORMAT_ERROR;
  case NM_IR_OP_UNSQUEEZE: {
    /* Axes count positions in the output. */
    int64_t axes[MAX_DIMS];
    size_t n = 0;
    if (!json_i64s(json_get(info->attrs, "axes"), axes, MAX_DIMS, &n) ||
        rank + n > MAX_DIMS)
      return NM_STATUS_UNSUPPORTED_TYPE;
    return axes_mask(info->attrs, rank + n, &node->cfg.axes.axes)
               ? NM_STATUS_OK
               : NM_STATUS_FORMAT_ERROR;
  }
  case NM_IR_OP_RANGE: {
    double limit = 0.0;
    if (!number_attr(info->attrs, "start", &node->cfg.range.start) ||
        !number_attr(info->attrs, "limit", &limit) ||
        !number_attr(info->attrs, "delta", &node->cfg.range.delta) ||
        node->cfg.range.delta == 0.0 || output->ndim != 1)
      return NM_STATUS_FORMAT_ERROR;
    double count = ceil((limit - node->cfg.range.start) / node->cfg.range.delta);
    if ((count > 0.0 ? count : 0.0) != (double)output->dims[0])
      return NM_STATUS_FORMAT_ERROR;
    return NM_STATUS_OK;
  }
  default:
    return NM_STATUS_OK; /* Equal, Not, Where have no attributes */
  }
}

static NmStatus read_node_config(const NodeInfo *info, NmIrNode *node,
                                 const Graph *source) {
  size_t a[MAX_DIMS], b[4], c[2], n = 0;
  const size_t input_rank =
      info->input_count ? source->values[info->inputs[0]].ndim : 0;
  if (node->op_type >= NM_IR_OP_GATHER)
    return read_tensor_op_config(info, node, source);
  if (node->op_type == NM_IR_OP_TRANSPOSE) {
    bool seen[MAX_DIMS] = {false};
    if (!int_attr(info->attrs, "perm", a, MAX_DIMS, &n) || n != input_rank)
      return NM_STATUS_UNSUPPORTED_TYPE;
    for (size_t i = 0; i < n; ++i) {
      if (a[i] >= n || seen[a[i]])
        return NM_STATUS_FORMAT_ERROR;
      seen[a[i]] = true;
      node->cfg.transpose.perm[i] = a[i];
    }
  }
  if (node->op_type == NM_IR_OP_LAYERNORM) {
    const Json *epsilon = json_get(info->attrs, "epsilon");
    if (!int_attr(info->attrs, "normalized_shape", a, MAX_DIMS, &n) || n == 0 ||
        n > input_rank || info->input_count > 3 ||
        (epsilon && (epsilon->type != J_NUM || !(epsilon->num > 0))))
      return NM_STATUS_UNSUPPORTED_TYPE;
    node->cfg.layernorm.norm_rank = n;
    node->cfg.layernorm.epsilon = epsilon ? (float)epsilon->num : 1e-5f;
  }
  if (node->op_type == NM_IR_OP_GELU) {
    const Json *approximate = json_get(info->attrs, "approximate");
    if (approximate && (approximate->type != J_STR ||
                        (strcmp(approximate->str, "tanh") &&
                         strcmp(approximate->str, "none"))))
      return NM_STATUS_UNSUPPORTED_TYPE;
    node->cfg.gelu.exact = approximate && !strcmp(approximate->str, "none");
  }
  if (node->op_type == NM_IR_OP_SOFTMAX) {
    /* The kernel normalizes the last axis only. */
    const Json *axis = json_get(info->attrs, "axis");
    if (axis && (axis->type != J_NUM ||
                 (axis->num != -1.0 && axis->num != (double)input_rank - 1.0)))
      return NM_STATUS_UNSUPPORTED_TYPE;
  }
  if (node->op_type == NM_IR_OP_CONV2D) {
    const Json *groups = json_get(info->attrs, "groups");
    if (!int_attr(info->attrs, "strides", a, 2, &n) || n != 2 ||
        !int_attr(info->attrs, "pads", b, 4, &n) || n != 4 ||
        !int_attr(info->attrs, "dilations", c, 2, &n) || n != 2 ||
        (groups && (groups->type != J_NUM || groups->num != 1)) ||
        c[0] != 1 || c[1] != 1 || b[0] != b[2] || b[1] != b[3])
      return NM_STATUS_UNSUPPORTED_TYPE;
    node->cfg.conv2d.stride_h = a[0];
    node->cfg.conv2d.stride_w = a[1];
    node->cfg.conv2d.pad_h = b[0];
    node->cfg.conv2d.pad_w = b[1];
    node->cfg.conv2d.algo = NM_CONV_ALGO_AUTO;
  } else if (node->op_type == NM_IR_OP_MAXPOOL2D) {
    if (!int_attr(info->attrs, "kernel", a, 2, &n) || n != 2 ||
        !int_attr(info->attrs, "strides", c, 2, &n) || n != 2 ||
        !int_attr(info->attrs, "pads", b, 4, &n) || n != 4 || b[0] != b[2] ||
        b[1] != b[3])
      return NM_STATUS_UNSUPPORTED_TYPE;
    node->cfg.maxpool2d.kernel_h = a[0];
    node->cfg.maxpool2d.kernel_w = a[1];
    node->cfg.maxpool2d.stride_h = c[0];
    node->cfg.maxpool2d.stride_w = c[1];
    node->cfg.maxpool2d.pad_h = b[0];
    node->cfg.maxpool2d.pad_w = b[1];
  }
  return NM_STATUS_OK;
}

static NmStatus build_model(const Json *root, const Graph *source,
                            PackageCache *cache, NmIrGraph *graph) {
  size_t input_ids[NM_IR_MAX_INPUTS], output_ids[2], n_in = 0, n_out = 0;
  const Json *inputs = json_get(root, "inputs");
  if (inputs && inputs->type == J_ARR && inputs->count > NM_IR_MAX_INPUTS)
    return NM_STATUS_UNSUPPORTED_TYPE;
  if (!json_ints(inputs, input_ids, NM_IR_MAX_INPUTS, &n_in) || n_in == 0 ||
      !json_ints(json_get(root, "outputs"), output_ids, 2, &n_out) ||
      n_out != 1 || output_ids[0] >= source->value_count)
    return NM_STATUS_FORMAT_ERROR;
  for (size_t i = 0; i < n_in; ++i) {
    if (input_ids[i] >= source->value_count)
      return NM_STATUS_FORMAT_ERROR;
  }
  /* A graph with one input takes a batch of any size; graphs with several
   * inputs (token ids and a mask, say) run with their declared shapes. */
  const bool any_batch = n_in == 1;
  int float_dtype = -1;

  bool *is_constant = calloc(source->value_count, sizeof(bool));
  if (!is_constant)
    return NM_STATUS_ALLOC_FAILED;
  for (size_t i = 0; i < source->node_count; ++i) {
    if (!strcmp(source->nodes[i].op, "Constant"))
      is_constant[source->nodes[i].output] = true;
  }

  NmStatus status = NM_STATUS_OK;
  const Json *values = json_get(root, "values");
  for (size_t i = 0; i < source->value_count && status == NM_STATUS_OK; ++i) {
    const ValueInfo *info = &source->values[i];
    NmIrShape shape = {.ndim = info->ndim};
    memcpy(shape.dims, info->dims, sizeof(shape.dims));
    bool is_input = false;
    for (size_t j = 0; j < n_in; ++j)
      is_input = is_input || i == input_ids[j];
    NmIrValueKind kind = is_constant[i]       ? NM_IR_VALUE_CONSTANT
                         : is_input           ? NM_IR_VALUE_MODEL_INPUT
                         : i == output_ids[0] ? NM_IR_VALUE_MODEL_OUTPUT
                                              : NM_IR_VALUE_INTERMEDIATE;
    size_t declared_batch = 0;
    if (any_batch && kind != NM_IR_VALUE_CONSTANT && shape.ndim >= 2) {
      declared_batch = shape.dims[0];
      shape.dims[0] = 0; /* any batch size */
    }
    if (info->dtype < 0) {
      status = NM_STATUS_UNSUPPORTED_TYPE;
      break;
    }
    /* One float type per graph: 16-bit and f32 values would need Cast. */
    if (nm_ir_dtype_is_float((uint8_t)info->dtype)) {
      if (float_dtype >= 0 && info->dtype != float_dtype) {
        status = NM_STATUS_UNSUPPORTED_TYPE;
        break;
      }
      float_dtype = info->dtype;
    }
    const Json *name = json_get(&values->items[i], "name");
    uint32_t id = nm_ir_graph_add_value(
        graph, name && name->type == J_STR ? name->str : "", shape, kind);
    if (id == NM_IR_INVALID_ID) {
      status = NM_STATUS_ALLOC_FAILED;
    } else {
      graph->values[id].dtype = (uint8_t)info->dtype;
      graph->values[id].declared_batch = declared_batch;
      if (kind == NM_IR_VALUE_CONSTANT)
        status = attach_constant(&graph->values[id], info, cache);
    }
  }

  for (size_t i = 0; i < source->node_count && status == NM_STATUS_OK; ++i) {
    const NodeInfo *info = &source->nodes[i];
    NmIrNode node;
    memset(&node, 0, sizeof(node));
    NmIrOpType op;
    if (!nm_ir_op_from_name(info->op, &op) || info->input_count > NM_IR_MAX_NODE_INPUTS) {
      status = NM_STATUS_UNSUPPORTED_TYPE;
      break;
    }
    node.op_type = (uint8_t)op;
    node.input_count = (uint32_t)info->input_count;
    for (size_t j = 0; j < info->input_count; ++j)
      node.input_ids[j] = (uint32_t)info->inputs[j];
    node.output_id = (uint32_t)info->output;
    status = read_node_config(info, &node, source);
    uint32_t node_id = nm_ir_graph_add_node(graph, &node);
    if (status == NM_STATUS_OK && node_id == NM_IR_INVALID_ID)
      status = NM_STATUS_ALLOC_FAILED;
    if (status == NM_STATUS_OK)
      graph->values[info->output].producer_node_id = node_id;
  }
  free(is_constant);
  graph->model_input_id = (uint32_t)input_ids[0];
  for (size_t i = 0; i < n_in; ++i)
    graph->input_ids[i] = (uint32_t)input_ids[i];
  graph->input_count = n_in;
  graph->model_output_id = (uint32_t)output_ids[0];
  return status;
}

static char *read_file(const char *path, size_t *size) {
  FILE *file = fopen(path, "rb");
  if (!file)
    return NULL;
  char *text = NULL;
  long length = 0;
  if (fseek(file, 0, SEEK_END) == 0 && (length = ftell(file)) >= 0 &&
      fseek(file, 0, SEEK_SET) == 0) {
    text = malloc((size_t)length + 1);
    if (text && fread(text, 1, (size_t)length, file) == (size_t)length) {
      text[length] = '\0';
      *size = (size_t)length;
    } else {
      free(text);
      text = NULL;
    }
  }
  fclose(file);
  return text;
}

/* Returns a copy of the directory part of `path` ("." when there is none). */
static char *document_directory(const char *path) {
  const char *slash = strrchr(path, '/');
  size_t length = slash ? (size_t)(slash - path) : 0;
  char *directory = malloc(length ? length + 1 : 2);
  if (!directory)
    return NULL;
  if (length) {
    memcpy(directory, path, length);
    directory[length] = '\0';
  } else {
    strcpy(directory, slash ? "/" : ".");
  }
  return directory;
}

NmStatus nm_import_graphir_json(const char *path, NmGraphModel **out_graph) {
  if (!path || !out_graph)
    return NM_STATUS_INVALID_ARGUMENT;
  *out_graph = NULL;
  size_t size = 0;
  char *text = read_file(path, &size);
  if (!text)
    return NM_STATUS_IO_ERROR;

  char *directory = document_directory(path);
  if (!directory) {
    free(text);
    return NM_STATUS_ALLOC_FAILED;
  }
  Parser parser = {.cur = text, .end = text + size};
  Json root = parse_value(&parser, 0);
  Graph source = {0};
  NmStatus status = NM_STATUS_FORMAT_ERROR;
  const Json *version = json_get(&root, "format_version");
  if (!parser.failed && root.type == J_OBJ && version && version->type == J_NUM &&
      version->num >= 3 && version->num <= 4 && load_graph(&root, &source)) {
    NmIrGraph *graph = malloc(sizeof(*graph));
    if (!graph) {
      status = NM_STATUS_ALLOC_FAILED;
    } else {
      nm_ir_graph_init(graph);
      PackageCache cache = {.directory = directory};
      status = build_model(&root, &source, &cache, graph);
      package_cache_free(&cache);
      if (status == NM_STATUS_OK) {
        *out_graph = graph;
      } else {
        nm_graph_model_free(graph);
      }
    }
  }
  free(source.values);
  free(source.nodes);
  json_free_children(&root);
  free(directory);
  free(text);
  return status;
}

void nm_graph_model_free(NmGraphModel *graph) {
  if (!graph)
    return;
  nm_ir_graph_destroy(graph);
  free(graph);
}
