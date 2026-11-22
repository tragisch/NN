/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/*
 * Graph interpreter for inference. The execution plan runs layer chains
 * (Dense, Conv2D, MaxPool2D, Flatten) with the training kernels; graphs that
 * branch or use tensor operations beyond those layers (residual Add, MatMul of
 * two activations, Transpose, LayerNorm, GELU, token embeddings, attention
 * masks, ...) run here instead, node by node over tensors of rank 0 to 6.
 * Float values compute in f32; int64 and bool values (token ids, masks) keep
 * their type. Semantics follow GraphIR's reference executor.
 */

#include "nm_model_internal.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DIMS NM_IR_MAX_DIMS

enum { T_F32 = NM_IR_DTYPE_F32, T_I64 = NM_IR_DTYPE_I64, T_BOOL = NM_IR_DTYPE_BOOL };

static size_t shape_numel(const NmIrShape *shape) {
  size_t count = 1;
  for (size_t i = 0; i < shape->ndim; ++i)
    count *= shape->dims[i];
  return count;
}

/* Type a value has while the interpreter runs: floats compute in f32. */
static uint8_t runtime_dtype(uint8_t dtype) {
  return nm_ir_dtype_is_float(dtype) ? T_F32 : dtype;
}

static size_t elem_size(uint8_t dtype) {
  return dtype == T_I64 ? sizeof(int64_t) : dtype == T_BOOL ? 1 : sizeof(float);
}

/* Left-pads `shape` with ones to rank DIMS. */
static void pad_dims(const NmIrShape *shape, size_t out[DIMS]) {
  size_t offset = DIMS - shape->ndim;
  for (size_t i = 0; i < DIMS; ++i)
    out[i] = i < offset ? 1 : shape->dims[i - offset];
}

/* Element strides of `shape` padded to rank DIMS, zero along the dimensions
 * that broadcast to `d`. */
static void broadcast_strides(const NmIrShape *shape, const size_t d[DIMS],
                              ptrdiff_t out[DIMS]) {
  size_t padded[DIMS];
  pad_dims(shape, padded);
  ptrdiff_t running = 1;
  for (size_t i = DIMS; i-- > 0;) {
    out[i] = padded[i] == 1 && d[i] != 1 ? 0 : running;
    running *= (ptrdiff_t)padded[i];
  }
}

/* Walks a row-major index space of rank DIMS and keeps up to three operand
 * offsets in step, each with its own strides. walk_finish drops dimensions of
 * size one and merges dimensions every operand reads contiguously, so the
 * kernels loop over rows of w.d[w.rank - 1] elements and step between rows
 * with walk_next_row. */
typedef struct {
  size_t d[DIMS];
  size_t c[DIMS];
  ptrdiff_t stride[3][DIMS];
  ptrdiff_t offset[3];
  size_t operands;
  size_t rank;
} Walk;

static void walk_init(Walk *w, const NmIrShape *shape, size_t operands) {
  memset(w, 0, sizeof(*w));
  pad_dims(shape, w->d);
  w->operands = operands;
}

static void walk_finish(Walk *w) {
  size_t rank = 0;
  for (size_t i = 0; i < DIMS; ++i) {
    if (w->d[i] == 1)
      continue;
    bool merge = rank > 0;
    for (size_t j = 0; merge && j < w->operands; ++j)
      merge = w->stride[j][rank - 1] == w->stride[j][i] * (ptrdiff_t)w->d[i];
    if (merge) {
      w->d[rank - 1] *= w->d[i];
      for (size_t j = 0; j < w->operands; ++j)
        w->stride[j][rank - 1] = w->stride[j][i];
      continue;
    }
    w->d[rank] = w->d[i];
    for (size_t j = 0; j < w->operands; ++j)
      w->stride[j][rank] = w->stride[j][i];
    rank++;
  }
  if (rank == 0) {
    w->d[0] = 1;
    for (size_t j = 0; j < w->operands; ++j)
      w->stride[j][0] = 0;
    rank = 1;
  }
  w->rank = rank;
}

/* Elements per row, and the number of rows. */
static size_t walk_inner(const Walk *w) { return w->d[w->rank - 1]; }

static size_t walk_rows(const Walk *w) {
  size_t rows = 1;
  for (size_t i = 0; i + 1 < w->rank; ++i)
    rows *= w->d[i];
  return rows;
}

static void walk_next_row(Walk *w) {
  for (size_t k = w->rank - 1; k-- > 0;) {
    for (size_t j = 0; j < w->operands; ++j)
      w->offset[j] += w->stride[j][k];
    if (++w->c[k] < w->d[k])
      return;
    for (size_t j = 0; j < w->operands; ++j)
      w->offset[j] -= w->stride[j][k] * (ptrdiff_t)w->d[k];
    w->c[k] = 0;
  }
}

/* out[i] = in[offset of operand 0] for every element of the walk. */
static void walk_copy(Walk *w, const void *in, size_t elem, void *out) {
  walk_finish(w);
  const size_t inner = walk_inner(w), rows = walk_rows(w);
  const ptrdiff_t step = w->stride[0][w->rank - 1];
  uint8_t *dst = out;
  for (size_t r = 0; r < rows; ++r, walk_next_row(w), dst += inner * elem) {
    const uint8_t *src = (const uint8_t *)in + w->offset[0] * (ptrdiff_t)elem;
    if (step == 1) {
      memcpy(dst, src, inner * elem);
    } else if (elem == 4) {
      const float *from = (const float *)src;
      float *to = (float *)dst;
      for (size_t i = 0; i < inner; ++i)
        to[i] = from[(ptrdiff_t)i * step];
    } else if (elem == 8) {
      const int64_t *from = (const int64_t *)src;
      int64_t *to = (int64_t *)dst;
      for (size_t i = 0; i < inner; ++i)
        to[i] = from[(ptrdiff_t)i * step];
    } else {
      for (size_t i = 0; i < inner; ++i)
        dst[i] = src[(ptrdiff_t)i * step];
    }
  }
}

static bool is_binary(uint8_t op) {
  return op == NM_IR_OP_ADD || op == NM_IR_OP_SUB || op == NM_IR_OP_MUL ||
         op == NM_IR_OP_DIV;
}

static bool is_unary(uint8_t op) {
  return op == NM_IR_OP_RELU || op == NM_IR_OP_SIGMOID ||
         op == NM_IR_OP_TANH || op == NM_IR_OP_SOFTMAX || op == NM_IR_OP_GELU;
}

/* ---- Shape inference ---------------------------------------------------- */

static bool broadcast_shape(const NmIrShape *a, const NmIrShape *b,
                            NmIrShape *out) {
  out->ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  for (size_t i = 0; i < out->ndim; ++i) {
    size_t da = i < out->ndim - a->ndim ? 1 : a->dims[i - (out->ndim - a->ndim)];
    size_t db = i < out->ndim - b->ndim ? 1 : b->dims[i - (out->ndim - b->ndim)];
    if (da != db && da != 1 && db != 1)
      return false;
    out->dims[i] = da == 1 ? db : da;
  }
  return true;
}

static bool matmul_shape(const NmIrShape *a, const NmIrShape *b,
                         NmIrShape *out) {
  if (a->ndim < 2 || b->ndim < 2)
    return false;
  size_t k = a->dims[a->ndim - 1];
  if (b->dims[b->ndim - 2] != k)
    return false;
  NmIrShape batch_a = {.ndim = a->ndim - 2};
  NmIrShape batch_b = {.ndim = b->ndim - 2};
  memcpy(batch_a.dims, a->dims, batch_a.ndim * sizeof(size_t));
  memcpy(batch_b.dims, b->dims, batch_b.ndim * sizeof(size_t));
  if (!broadcast_shape(&batch_a, &batch_b, out) || out->ndim + 2 > DIMS)
    return false;
  out->dims[out->ndim++] = a->dims[a->ndim - 2];
  out->dims[out->ndim++] = b->dims[b->ndim - 1];
  return true;
}

static size_t pooled(size_t in, size_t kernel, size_t stride, size_t pad) {
  if (stride == 0 || in + 2 * pad < kernel)
    return 0;
  return (in + 2 * pad - kernel) / stride + 1;
}

/* Start and step of one sliced axis after GraphIR's clamping, and its length. */
static size_t slice_axis(size_t dim, int64_t start, int64_t end, int64_t step,
                         int64_t *first) {
  const int64_t d = (int64_t)dim;
  if (start < 0)
    start = (start < -d - 1 ? -d - 1 : start) + d;
  if (end < 0)
    end = (end < -d - 1 ? -d - 1 : end) + d;
  if (step > 0) {
    start = start < 0 ? 0 : start > d ? d : start;
    end = end < 0 ? 0 : end > d ? d : end;
    *first = start;
    return end <= start ? 0 : (size_t)((end - start - 1) / step + 1);
  }
  start = start < -1 ? -1 : start > d - 1 ? d - 1 : start;
  end = end < -1 ? -1 : end > d - 1 ? d - 1 : end;
  *first = start;
  return start <= end ? 0 : (size_t)((start - end - 1) / -step + 1);
}

static bool is_float_op(uint8_t op) {
  return is_unary(op) || is_binary(op) || op == NM_IR_OP_MATMUL ||
         op == NM_IR_OP_LAYERNORM || op == NM_IR_OP_CONV2D ||
         op == NM_IR_OP_MAXPOOL2D || op == NM_IR_OP_REDUCEMAX;
}

/* Output shape and type of `node` for the runtime input shapes and types. */
static bool infer_shape(const NmIrGraph *graph, const NmIrNode *node,
                        const NmIrShape *in, const uint8_t *in_t,
                        NmIrShape *out, uint8_t *out_t) {
  const uint8_t op = node->op_type;
  *out_t = node->input_count ? in_t[0] : T_F32;
  if (is_float_op(op)) {
    for (uint32_t i = 0; i < node->input_count; ++i) {
      if (in_t[i] != T_F32)
        return false;
    }
  }
  if (is_unary(op)) {
    *out = in[0];
    return node->input_count == 1;
  }
  if (is_binary(op))
    return node->input_count == 2 && broadcast_shape(&in[0], &in[1], out);
  switch (op) {
  case NM_IR_OP_MATMUL:
    return node->input_count == 2 && matmul_shape(&in[0], &in[1], out);
  case NM_IR_OP_TRANSPOSE: {
    if (node->input_count != 1)
      return false;
    out->ndim = in[0].ndim;
    bool seen[DIMS] = {false};
    for (size_t i = 0; i < out->ndim; ++i) {
      size_t axis = node->cfg.transpose.perm[i];
      if (axis >= in[0].ndim || seen[axis])
        return false;
      seen[axis] = true;
      out->dims[i] = in[0].dims[axis];
    }
    return true;
  }
  case NM_IR_OP_RESHAPE: {
    if (node->input_count != 1)
      return false;
    /* The declared shape, with the batch dimension (0) derived from the
     * element count so that reshapes may fold the batch into a dimension. */
    *out = graph->values[node->output_id].shape;
    size_t known = 1;
    bool open = false;
    for (size_t i = 0; i < out->ndim; ++i) {
      if (out->dims[i] == 0 && i == 0)
        open = true;
      else
        known *= out->dims[i];
    }
    size_t total = shape_numel(&in[0]);
    if (open) {
      if (known == 0 || total % known != 0)
        return false;
      out->dims[0] = total / known;
    }
    return shape_numel(out) == total;
  }
  case NM_IR_OP_LAYERNORM: {
    if (node->input_count < 1 || node->input_count > 3 ||
        node->cfg.layernorm.norm_rank == 0 ||
        node->cfg.layernorm.norm_rank > in[0].ndim)
      return false;
    size_t group = 1;
    for (size_t i = in[0].ndim - node->cfg.layernorm.norm_rank; i < in[0].ndim;
         ++i)
      group *= in[0].dims[i];
    for (uint32_t i = 1; i < node->input_count; ++i) {
      if (shape_numel(&in[i]) != group)
        return false;
    }
    *out = in[0];
    return true;
  }
  case NM_IR_OP_CONV2D: {
    if (node->input_count < 2 || node->input_count > 3 || in[0].ndim != 4 ||
        in[1].ndim != 4 || in[1].dims[1] != in[0].dims[1] ||
        (node->input_count == 3 && shape_numel(&in[2]) != in[1].dims[0]))
      return false;
    out->ndim = 4;
    out->dims[0] = in[0].dims[0];
    out->dims[1] = in[1].dims[0];
    out->dims[2] = pooled(in[0].dims[2], in[1].dims[2],
                          node->cfg.conv2d.stride_h, node->cfg.conv2d.pad_h);
    out->dims[3] = pooled(in[0].dims[3], in[1].dims[3],
                          node->cfg.conv2d.stride_w, node->cfg.conv2d.pad_w);
    return out->dims[2] && out->dims[3];
  }
  case NM_IR_OP_MAXPOOL2D:
    if (node->input_count != 1 || in[0].ndim != 4)
      return false;
    *out = in[0];
    out->dims[2] = pooled(in[0].dims[2], node->cfg.maxpool2d.kernel_h,
                          node->cfg.maxpool2d.stride_h, node->cfg.maxpool2d.pad_h);
    out->dims[3] = pooled(in[0].dims[3], node->cfg.maxpool2d.kernel_w,
                          node->cfg.maxpool2d.stride_w, node->cfg.maxpool2d.pad_w);
    return out->dims[2] && out->dims[3];
  case NM_IR_OP_GATHER: {
    size_t axis = node->cfg.gather.axis;
    if (node->input_count != 2 || in_t[1] != T_I64 || axis >= in[0].ndim ||
        in[0].ndim - 1 + in[1].ndim > DIMS)
      return false;
    out->ndim = 0;
    for (size_t i = 0; i < axis; ++i)
      out->dims[out->ndim++] = in[0].dims[i];
    for (size_t i = 0; i < in[1].ndim; ++i)
      out->dims[out->ndim++] = in[1].dims[i];
    for (size_t i = axis + 1; i < in[0].ndim; ++i)
      out->dims[out->ndim++] = in[0].dims[i];
    return true;
  }
  case NM_IR_OP_EQUAL:
    *out_t = T_BOOL;
    return node->input_count == 2 && in_t[0] == in_t[1] &&
           broadcast_shape(&in[0], &in[1], out);
  case NM_IR_OP_NOT:
    *out = in[0];
    return node->input_count == 1 && in_t[0] == T_BOOL;
  case NM_IR_OP_WHERE: {
    NmIrShape partial;
    *out_t = in_t[1];
    return node->input_count == 3 && in_t[0] == T_BOOL && in_t[1] == in_t[2] &&
           broadcast_shape(&in[0], &in[1], &partial) &&
           broadcast_shape(&partial, &in[2], out);
  }
  case NM_IR_OP_CAST:
    *out = in[0];
    *out_t = node->cfg.cast.to;
    return node->input_count == 1;
  case NM_IR_OP_EXPAND: {
    NmIrShape target = {.ndim = node->cfg.expand.ndim};
    memcpy(target.dims, node->cfg.expand.dims, sizeof(target.dims));
    return node->input_count == 1 && broadcast_shape(&in[0], &target, out);
  }
  case NM_IR_OP_SLICE: {
    if (node->input_count != 1)
      return false;
    out->ndim = in[0].ndim;
    for (size_t i = 0; i < in[0].ndim; ++i) {
      int64_t first;
      out->dims[i] = slice_axis(in[0].dims[i], node->cfg.slice.starts[i],
                                node->cfg.slice.ends[i],
                                node->cfg.slice.steps[i], &first);
    }
    return true;
  }
  case NM_IR_OP_REDUCEMAX: {
    if (node->input_count != 1 || node->cfg.axes.axes >> in[0].ndim)
      return false;
    out->ndim = 0;
    for (size_t i = 0; i < in[0].ndim; ++i) {
      bool reduced = node->cfg.axes.axes & (1u << i);
      if (reduced && in[0].dims[i] == 0)
        return false;
      if (!reduced)
        out->dims[out->ndim++] = in[0].dims[i];
      else if (node->cfg.axes.keepdims)
        out->dims[out->ndim++] = 1;
    }
    return true;
  }
  case NM_IR_OP_RANGE:
    *out = graph->values[node->output_id].shape;
    *out_t = runtime_dtype(graph->values[node->output_id].dtype);
    return node->input_count == 0 && out->ndim == 1 &&
           (*out_t == T_F32 || *out_t == T_I64);
  case NM_IR_OP_SQUEEZE: {
    if (node->input_count != 1 || node->cfg.axes.axes >> in[0].ndim)
      return false;
    out->ndim = 0;
    for (size_t i = 0; i < in[0].ndim; ++i) {
      if (!(node->cfg.axes.axes & (1u << i)))
        out->dims[out->ndim++] = in[0].dims[i];
      else if (in[0].dims[i] != 1)
        return false;
    }
    return true;
  }
  case NM_IR_OP_UNSQUEEZE: {
    if (node->input_count != 1)
      return false;
    size_t inserted = 0, next = 0;
    for (size_t i = 0; i < DIMS; ++i)
      inserted += (node->cfg.axes.axes >> i) & 1u;
    out->ndim = in[0].ndim + inserted;
    if (out->ndim > DIMS || node->cfg.axes.axes >> out->ndim)
      return false;
    for (size_t i = 0; i < out->ndim; ++i)
      out->dims[i] =
          node->cfg.axes.axes & (1u << i) ? 1 : in[0].dims[next++];
    return true;
  }
  default:
    return false;
  }
}

/* ---- Kernels ------------------------------------------------------------ */

static float apply_binary(uint8_t op, float a, float b) {
  switch (op) {
  case NM_IR_OP_ADD: return a + b;
  case NM_IR_OP_SUB: return a - b;
  case NM_IR_OP_MUL: return a * b;
  default: return a / b;
  }
}

/* Row kernels with the operation hoisted out of the loop, so they vectorize.
 * `sa`/`sb` are 1 for a row and 0 for a value repeated along it. */
#define BINARY_ROW(NAME, A, B)                                                 \
  static void NAME(uint8_t op, const float *a, const float *b, float *out,    \
                   size_t n) {                                                 \
    switch (op) {                                                              \
    case NM_IR_OP_ADD:                                                         \
      for (size_t i = 0; i < n; ++i)                                           \
        out[i] = (A) + (B);                                                    \
      break;                                                                   \
    case NM_IR_OP_SUB:                                                         \
      for (size_t i = 0; i < n; ++i)                                           \
        out[i] = (A) - (B);                                                    \
      break;                                                                   \
    case NM_IR_OP_MUL:                                                         \
      for (size_t i = 0; i < n; ++i)                                           \
        out[i] = (A) * (B);                                                    \
      break;                                                                   \
    default:                                                                   \
      for (size_t i = 0; i < n; ++i)                                           \
        out[i] = (A) / (B);                                                    \
      break;                                                                   \
    }                                                                          \
  }

BINARY_ROW(binary_row, a[i], b[i])
BINARY_ROW(binary_row_scalar_b, a[i], b[0])
BINARY_ROW(binary_row_scalar_a, a[0], b[i])

static void run_binary(uint8_t op, const float *a, const NmIrShape *sa,
                       const float *b, const NmIrShape *sb, float *out,
                       const NmIrShape *so) {
  size_t n = shape_numel(so);
  size_t na = shape_numel(sa), nb = shape_numel(sb);
  if (na == n && nb == n) {
    binary_row(op, a, b, out, n);
    return;
  }
  if (na == n && nb == 1) {
    binary_row_scalar_b(op, a, b, out, n);
    return;
  }
  if (na == n && nb > 0 && n % nb == 0 && sb->ndim <= so->ndim) {
    /* b broadcasts along leading dimensions only (bias, scale): b's shape is
     * a suffix of the output shape. */
    bool suffix = true;
    for (size_t i = 0; i < sb->ndim; ++i)
      suffix = suffix && sb->dims[i] == so->dims[so->ndim - sb->ndim + i];
    if (suffix) {
      for (size_t i = 0; i < n; i += nb)
        binary_row(op, a + i, b, out + i, nb);
      return;
    }
  }
  Walk w;
  walk_init(&w, so, 2);
  broadcast_strides(sa, w.d, w.stride[0]);
  broadcast_strides(sb, w.d, w.stride[1]);
  walk_finish(&w);
  const size_t inner = walk_inner(&w), rows = walk_rows(&w);
  const ptrdiff_t ta = w.stride[0][w.rank - 1], tb = w.stride[1][w.rank - 1];
  for (size_t r = 0; r < rows; ++r, walk_next_row(&w), out += inner) {
    const float *pa = a + w.offset[0], *pb = b + w.offset[1];
    if (ta == 1 && tb == 1)
      binary_row(op, pa, pb, out, inner);
    else if (ta == 1 && tb == 0)
      binary_row_scalar_b(op, pa, pb, out, inner);
    else if (ta == 0 && tb == 1)
      binary_row_scalar_a(op, pa, pb, out, inner);
    else
      for (size_t i = 0; i < inner; ++i)
        out[i] = apply_binary(op, pa[(ptrdiff_t)i * ta], pb[(ptrdiff_t)i * tb]);
  }
}

static float gelu_tanh(float x) {
  const float k = 0.7978846f; /* sqrt(2 / pi) */
  return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

/* e^x in f32 without branches on the data, so loops over it vectorize
 * (Cephes expf polynomial, within about 2 ulp; GraphIR's exp_f32). */
static inline float exp_fast(float x) {
  const float clamped = x > 88.0f ? 88.0f : x < -88.0f ? -88.0f : x;
  const float n = roundf(clamped * 1.44269504f);
  const float r = clamped - n * 0.693359375f - n * -2.12194440e-4f;
  float series = 1.9875691e-4f;
  series = series * r + 1.3981999e-3f;
  series = series * r + 8.3334520e-3f;
  series = series * r + 4.1665796e-2f;
  series = series * r + 0.16666665f;
  series = series * r + 0.5f;
  union {
    uint32_t bits;
    float value;
  } scale = {.bits = (uint32_t)((int32_t)n + 127) << 23};
  const float y = (series * r * r + r + 1.0f) * scale.value;
  return x < -87.3f ? 0.0f : y;
}

/* erf with an absolute error below 3e-7 (Numerical Recipes erfc
 * polynomial, as GraphIR's reference executor computes it). */
static inline float erf_fast(float x) {
  const float z = fabsf(x);
  const float t = 1.0f / (1.0f + 0.5f * z);
  float p = 0.17087277f;
  p = p * t - 0.82215223f;
  p = p * t + 1.48851587f;
  p = p * t - 1.13520398f;
  p = p * t + 0.27886807f;
  p = p * t - 0.18628806f;
  p = p * t + 0.09678418f;
  p = p * t + 0.37409196f;
  p = p * t + 1.00002368f;
  p = p * t - 1.26551223f;
  const float e = 1.0f - t * exp_fast(p - z * z);
  return copysignf(e, x);
}

static float gelu_erf(float x) {
  return 0.5f * x * (1.0f + erf_fast(x * 0.70710678f));
}

static void run_unary(const NmIrNode *node, const float *in, float *out,
                      const NmIrShape *shape) {
  size_t n = shape_numel(shape);
  switch (node->op_type) {
  case NM_IR_OP_RELU:
    for (size_t i = 0; i < n; ++i)
      out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    break;
  case NM_IR_OP_SIGMOID:
    for (size_t i = 0; i < n; ++i)
      out[i] = 1.0f / (1.0f + expf(-in[i]));
    break;
  case NM_IR_OP_TANH:
    for (size_t i = 0; i < n; ++i)
      out[i] = tanhf(in[i]);
    break;
  case NM_IR_OP_GELU:
    if (node->cfg.gelu.exact) {
      /* Several vectors at once hide the latency of the polynomials. */
#pragma clang loop interleave_count(4)
      for (size_t i = 0; i < n; ++i)
        out[i] = gelu_erf(in[i]);
    } else {
      for (size_t i = 0; i < n; ++i)
        out[i] = gelu_tanh(in[i]);
    }
    break;
  case NM_IR_OP_SOFTMAX: {
    size_t cols = shape->dims[shape->ndim - 1];
    for (size_t row = 0; cols && row < n / cols; ++row) {
      const float *x = in + row * cols;
      float *y = out + row * cols;
      float max = x[0];
      for (size_t j = 1; j < cols; ++j)
        max = fmaxf(max, x[j]);
      float sum = 0.0f;
      for (size_t j = 0; j < cols; ++j)
        y[j] = exp_fast(x[j] - max);
      for (size_t j = 0; j < cols; ++j)
        sum += y[j];
      const float inverse = 1.0f / sum;
      for (size_t j = 0; j < cols; ++j)
        y[j] *= inverse;
    }
    break;
  }
  default:
    break;
  }
}

static bool gemm(const float *a, const float *b, float *c, size_t m, size_t k,
                 size_t n) {
  FloatMatrix ma = {.rows = m, .cols = k, .capacity = m * k, .values = (float *)a};
  FloatMatrix mb = {.rows = k, .cols = n, .capacity = k * n, .values = (float *)b};
  FloatMatrix mc = {.rows = m, .cols = n, .capacity = m * n, .values = c};
  return sm_gemm(&mc, 1.0f, &ma, SM_NO_TRANSPOSE, &mb, SM_NO_TRANSPOSE, 0.0f);
}

static bool run_matmul(const float *a, const NmIrShape *sa, const float *b,
                       const NmIrShape *sb, float *out, const NmIrShape *so) {
  size_t m = sa->dims[sa->ndim - 2], k = sa->dims[sa->ndim - 1];
  size_t n = sb->dims[sb->ndim - 1];
  if (shape_numel(sb) == k * n) {
    /* b is one matrix shared by every batch: a's batches stack into the rows
     * of one product (a projection of [batch, tokens, features]). */
    return gemm(a, b, out, shape_numel(sa) / k, k, n);
  }
  NmIrShape ba = {.ndim = sa->ndim - 2}, bb = {.ndim = sb->ndim - 2},
            bo = {.ndim = so->ndim - 2};
  memcpy(ba.dims, sa->dims, ba.ndim * sizeof(size_t));
  memcpy(bb.dims, sb->dims, bb.ndim * sizeof(size_t));
  memcpy(bo.dims, so->dims, bo.ndim * sizeof(size_t));
  size_t da[DIMS], db[DIMS], d[DIMS];
  pad_dims(&ba, da);
  pad_dims(&bb, db);
  pad_dims(&bo, d);
  size_t batches = shape_numel(&bo);
  for (size_t index = 0; index < batches; ++index) {
    /* Decompose the output batch index and map it onto a and b. */
    size_t rest = index, offset_a = 0, offset_b = 0, scale_a = 1, scale_b = 1;
    for (size_t i = DIMS; i-- > 0;) {
      size_t coordinate = rest % d[i];
      rest /= d[i];
      offset_a += (da[i] == 1 ? 0 : coordinate) * scale_a;
      offset_b += (db[i] == 1 ? 0 : coordinate) * scale_b;
      scale_a *= da[i];
      scale_b *= db[i];
    }
    if (!gemm(a + offset_a * m * k, b + offset_b * k * n, out + index * m * n,
              m, k, n))
      return false;
  }
  return true;
}

/* Sum of `n` floats with eight partial sums, so the loop vectorizes. */
static float sum8(const float *x, size_t n) {
  float lanes[8] = {0.0f};
  size_t j = 0;
  for (; j + 8 <= n; j += 8)
    for (size_t l = 0; l < 8; ++l)
      lanes[l] += x[j + l];
  float sum = 0.0f;
  for (; j < n; ++j)
    sum += x[j];
  for (size_t l = 0; l < 8; ++l)
    sum += lanes[l];
  return sum;
}

static void run_layernorm(const NmIrNode *node, const float *in,
                          const NmIrShape *shape, const float *scale,
                          const float *bias, float *out) {
  size_t group = 1;
  for (size_t i = shape->ndim - node->cfg.layernorm.norm_rank; i < shape->ndim;
       ++i)
    group *= shape->dims[i];
  size_t groups = shape_numel(shape) / group;
  for (size_t g = 0; g < groups; ++g) {
    const float *x = in + g * group;
    float *y = out + g * group;
    const float mean = sum8(x, group) / (float)group;
    /* y holds the centered values first. */
    for (size_t j = 0; j < group; ++j)
      y[j] = x[j] - mean;
    float lanes[8] = {0.0f};
    size_t j = 0;
    for (; j + 8 <= group; j += 8)
      for (size_t l = 0; l < 8; ++l)
        lanes[l] += y[j + l] * y[j + l];
    float variance = 0.0f;
    for (; j < group; ++j)
      variance += y[j] * y[j];
    for (size_t l = 0; l < 8; ++l)
      variance += lanes[l];
    variance /= (float)group;
    const float inverse = 1.0f / sqrtf(variance + node->cfg.layernorm.epsilon);
    if (scale && bias) {
      for (size_t i = 0; i < group; ++i)
        y[i] = y[i] * inverse * scale[i] + bias[i];
    } else {
      for (size_t i = 0; i < group; ++i) {
        float value = y[i] * inverse;
        if (scale)
          value *= scale[i];
        if (bias)
          value += bias[i];
        y[i] = value;
      }
    }
  }
}

static void run_conv2d_naive(const NmIrNode *node, const float *in,
                       const NmIrShape *si, const float *weight,
                       const NmIrShape *sw, const float *bias, float *out,
                       const NmIrShape *so) {
  size_t batch = si->dims[0], channels = si->dims[1], height = si->dims[2],
         width = si->dims[3];
  size_t filters = sw->dims[0], kh = sw->dims[2], kw = sw->dims[3];
  size_t out_h = so->dims[2], out_w = so->dims[3];
  size_t sh = node->cfg.conv2d.stride_h, sw_ = node->cfg.conv2d.stride_w;
  size_t ph = node->cfg.conv2d.pad_h, pw = node->cfg.conv2d.pad_w;
  for (size_t n = 0; n < batch; ++n)
    for (size_t f = 0; f < filters; ++f)
      for (size_t oy = 0; oy < out_h; ++oy)
        for (size_t ox = 0; ox < out_w; ++ox) {
          float sum = bias ? bias[f] : 0.0f;
          for (size_t c = 0; c < channels; ++c)
            for (size_t ky = 0; ky < kh; ++ky) {
              size_t y = oy * sh + ky;
              if (y < ph || y - ph >= height)
                continue;
              for (size_t kx = 0; kx < kw; ++kx) {
                size_t x = ox * sw_ + kx;
                if (x < pw || x - pw >= width)
                  continue;
                sum += in[((n * channels + c) * height + (y - ph)) * width +
                          (x - pw)] *
                       weight[((f * channels + c) * kh + ky) * kw + kx];
              }
            }
          out[((n * filters + f) * out_h + oy) * out_w + ox] = sum;
        }
}

static void run_maxpool2d_naive(const NmIrNode *node, const float *in,
                          const NmIrShape *si, float *out,
                          const NmIrShape *so) {
  size_t planes = si->dims[0] * si->dims[1], height = si->dims[2],
         width = si->dims[3];
  size_t out_h = so->dims[2], out_w = so->dims[3];
  for (size_t p = 0; p < planes; ++p)
    for (size_t oy = 0; oy < out_h; ++oy)
      for (size_t ox = 0; ox < out_w; ++ox) {
        float max = -INFINITY;
        for (size_t ky = 0; ky < node->cfg.maxpool2d.kernel_h; ++ky) {
          size_t y = oy * node->cfg.maxpool2d.stride_h + ky;
          if (y < node->cfg.maxpool2d.pad_h ||
              y - node->cfg.maxpool2d.pad_h >= height)
            continue;
          for (size_t kx = 0; kx < node->cfg.maxpool2d.kernel_w; ++kx) {
            size_t x = ox * node->cfg.maxpool2d.stride_w + kx;
            if (x < node->cfg.maxpool2d.pad_w ||
                x - node->cfg.maxpool2d.pad_w >= width)
              continue;
            float v = in[(p * height + (y - node->cfg.maxpool2d.pad_h)) * width +
                         (x - node->cfg.maxpool2d.pad_w)];
            max = v > max ? v : max;
          }
        }
        out[(p * out_h + oy) * out_w + ox] = max;
      }
}


/* Borrowed view of an interpreter buffer as an mMatrix tensor. */
static FloatTensor *tensor_view(const float *data, const NmIrShape *shape) {
  return st_create_with_data(shape->ndim, shape->dims, (float *)data,
                             shape_numel(shape), false);
}

/* Convolution and pooling run on mMatrix kernels (GEMM, Metal for large
 * sizes); the reference loops above remain for anything they reject. */
static void run_conv2d(const NmIrNode *node, const float *in,
                       const NmIrShape *si, const float *weight,
                       const NmIrShape *sw, const float *bias, float *out,
                       const NmIrShape *so) {
  FloatTensor *x = tensor_view(in, si), *w = tensor_view(weight, sw);
  FloatTensor *y = tensor_view(out, so);
  NmIrShape bias_shape = {.ndim = 1, .dims = {sw->dims[0]}};
  FloatTensor *b = bias ? tensor_view(bias, &bias_shape) : NULL;
  StConv2dParams params = st_conv2d_default_params();
  params.stride_h = node->cfg.conv2d.stride_h;
  params.stride_w = node->cfg.conv2d.stride_w;
  params.pad_h = node->cfg.conv2d.pad_h;
  params.pad_w = node->cfg.conv2d.pad_w;
  bool ok = x && w && y && (!bias || b) &&
            st_conv2d_nchw(x, w, b, &params, y);
  st_destroy(x);
  st_destroy(w);
  st_destroy(y);
  st_destroy(b);
  if (!ok)
    run_conv2d_naive(node, in, si, weight, sw, bias, out, so);
}

static void run_maxpool2d(const NmIrNode *node, const float *in,
                          const NmIrShape *si, float *out,
                          const NmIrShape *so) {
  FloatTensor *x = tensor_view(in, si), *y = tensor_view(out, so);
  bool ok = x && y &&
            st_maxpool2d_nchw(x, node->cfg.maxpool2d.kernel_h,
                              node->cfg.maxpool2d.kernel_w,
                              node->cfg.maxpool2d.stride_h,
                              node->cfg.maxpool2d.stride_w,
                              node->cfg.maxpool2d.pad_h,
                              node->cfg.maxpool2d.pad_w, y, NULL);
  st_destroy(x);
  st_destroy(y);
  if (!ok)
    run_maxpool2d_naive(node, in, si, out, so);
}

static void run_transpose(const NmIrNode *node, const void *in,
                          const NmIrShape *si, size_t elem, void *out,
                          const NmIrShape *so) {
  /* Stride in the input of each output axis. */
  size_t in_stride[DIMS];
  size_t running = 1;
  for (size_t i = si->ndim; i-- > 0;) {
    in_stride[i] = running;
    running *= si->dims[i];
  }
  Walk w;
  walk_init(&w, so, 1);
  size_t offset = DIMS - so->ndim;
  for (size_t i = offset; i < DIMS; ++i)
    w.stride[0][i] = (ptrdiff_t)in_stride[node->cfg.transpose.perm[i - offset]];
  walk_copy(&w, in, elem, out);
}

/* Gathers slices along the axis; negative indices count from the end. */
static bool run_gather(const NmIrNode *node, const void *data,
                       const NmIrShape *sd, size_t elem,
                       const int64_t *indices, size_t count, void *out) {
  size_t axis = node->cfg.gather.axis, outer = 1, inner = 1;
  for (size_t i = 0; i < axis; ++i)
    outer *= sd->dims[i];
  for (size_t i = axis + 1; i < sd->ndim; ++i)
    inner *= sd->dims[i];
  const int64_t dim = (int64_t)sd->dims[axis];
  const size_t bytes = inner * elem;
  uint8_t *dst = out;
  for (size_t o = 0; o < outer; ++o)
    for (size_t j = 0; j < count; ++j) {
      int64_t index = indices[j] < 0 ? indices[j] + dim : indices[j];
      if (index < 0 || index >= dim)
        return false;
      memcpy(dst, (const uint8_t *)data + (o * (size_t)dim + (size_t)index) * bytes,
             bytes);
      dst += bytes;
    }
  return true;
}

static void run_equal(uint8_t dtype, const void *a, const NmIrShape *sa,
                      const void *b, const NmIrShape *sb, uint8_t *out,
                      const NmIrShape *so) {
  Walk w;
  walk_init(&w, so, 2);
  broadcast_strides(sa, w.d, w.stride[0]);
  broadcast_strides(sb, w.d, w.stride[1]);
  walk_finish(&w);
  const size_t inner = walk_inner(&w), rows = walk_rows(&w);
  const ptrdiff_t ta = w.stride[0][w.rank - 1], tb = w.stride[1][w.rank - 1];
  for (size_t r = 0; r < rows; ++r, walk_next_row(&w), out += inner) {
    const ptrdiff_t ia = w.offset[0], ib = w.offset[1];
    for (size_t i = 0; i < inner; ++i) {
      const ptrdiff_t xa = ia + (ptrdiff_t)i * ta, xb = ib + (ptrdiff_t)i * tb;
      switch (dtype) {
      case T_F32: out[i] = ((const float *)a)[xa] == ((const float *)b)[xb]; break;
      case T_I64: out[i] = ((const int64_t *)a)[xa] == ((const int64_t *)b)[xb]; break;
      default:
        out[i] = (((const uint8_t *)a)[xa] != 0) == (((const uint8_t *)b)[xb] != 0);
        break;
      }
    }
  }
}

static void run_where(const uint8_t *cond, const NmIrShape *sc, const void *x,
                      const NmIrShape *sx, const void *y, const NmIrShape *sy,
                      size_t elem, void *out, const NmIrShape *so) {
  Walk w;
  walk_init(&w, so, 3);
  broadcast_strides(sc, w.d, w.stride[0]);
  broadcast_strides(sx, w.d, w.stride[1]);
  broadcast_strides(sy, w.d, w.stride[2]);
  walk_finish(&w);
  const size_t inner = walk_inner(&w), rows = walk_rows(&w);
  const ptrdiff_t tc = w.stride[0][w.rank - 1], tx = w.stride[1][w.rank - 1],
                  ty = w.stride[2][w.rank - 1];
  if (elem == 4) {
    const float *fx = x, *fy = y;
    float *o = out;
    for (size_t r = 0; r < rows; ++r, walk_next_row(&w), o += inner) {
      const uint8_t *c = cond + w.offset[0];
      const float *px = fx + w.offset[1], *py = fy + w.offset[2];
      if (tc == 0) {
        /* One condition for the whole row: copy the chosen row. */
        const float *from = *c ? px : py;
        const ptrdiff_t step = *c ? tx : ty;
        if (step == 1)
          memcpy(o, from, inner * sizeof(float));
        else
          for (size_t i = 0; i < inner; ++i)
            o[i] = from[(ptrdiff_t)i * step];
      } else {
        for (size_t i = 0; i < inner; ++i) {
          const ptrdiff_t k = (ptrdiff_t)i;
          o[i] = c[k * tc] ? px[k * tx] : py[k * ty];
        }
      }
    }
    return;
  }
  uint8_t *dst = out;
  for (size_t r = 0; r < rows; ++r, walk_next_row(&w)) {
    for (size_t i = 0; i < inner; ++i, dst += elem) {
      const ptrdiff_t k = (ptrdiff_t)i;
      const uint8_t *from =
          cond[w.offset[0] + k * tc]
              ? (const uint8_t *)x + (w.offset[1] + k * tx) * (ptrdiff_t)elem
              : (const uint8_t *)y + (w.offset[2] + k * ty) * (ptrdiff_t)elem;
      memcpy(dst, from, elem);
    }
  }
}

/* C-style conversions: floats truncate toward zero (saturating, NaN becomes
 * 0), non-zero values become true, booleans become 0 or 1. */
static int64_t float_to_i64(float value) {
  if (isnan(value))
    return 0;
  if (value >= 9.2233720368547758e18f)
    return INT64_MAX;
  if (value <= -9.2233720368547758e18f)
    return INT64_MIN;
  return (int64_t)value;
}

static void run_cast(uint8_t from, const void *in, uint8_t to, void *out,
                     size_t n) {
  for (size_t i = 0; i < n; ++i) {
    double value = 0.0;
    int64_t integer = 0;
    bool truth = false;
    switch (from) {
    case T_F32: {
      float f = ((const float *)in)[i];
      value = f;
      integer = float_to_i64(f);
      truth = f != 0.0f;
      break;
    }
    case T_I64:
      integer = ((const int64_t *)in)[i];
      value = (double)integer;
      truth = integer != 0;
      break;
    default:
      truth = ((const uint8_t *)in)[i] != 0;
      integer = truth;
      value = truth;
      break;
    }
    switch (to) {
    case T_F32: ((float *)out)[i] = (float)value; break;
    case T_I64: ((int64_t *)out)[i] = integer; break;
    default: ((uint8_t *)out)[i] = truth; break;
    }
  }
}

static void run_slice(const NmIrNode *node, const void *in,
                      const NmIrShape *si, size_t elem, void *out,
                      const NmIrShape *so) {
  Walk w;
  walk_init(&w, so, 1);
  size_t offset = DIMS - si->ndim;
  ptrdiff_t running = 1;
  for (size_t i = si->ndim; i-- > 0;) {
    int64_t first;
    slice_axis(si->dims[i], node->cfg.slice.starts[i], node->cfg.slice.ends[i],
               node->cfg.slice.steps[i], &first);
    w.offset[0] += (ptrdiff_t)first * running;
    w.stride[0][i + offset] = (ptrdiff_t)node->cfg.slice.steps[i] * running;
    running *= (ptrdiff_t)si->dims[i];
  }
  walk_copy(&w, in, elem, out);
}

static void run_reducemax(const NmIrNode *node, const float *in,
                          const NmIrShape *si, float *out,
                          const NmIrShape *so) {
  /* Walk the input; the output offset stands still along reduced axes. */
  NmIrShape kept = *si;
  for (size_t i = 0; i < si->ndim; ++i) {
    if (node->cfg.axes.axes & (1u << i))
      kept.dims[i] = 1;
  }
  Walk w;
  walk_init(&w, si, 2);
  broadcast_strides(&kept, w.d, w.stride[0]);
  broadcast_strides(si, w.d, w.stride[1]);
  walk_finish(&w);
  for (size_t i = 0; i < shape_numel(so); ++i)
    out[i] = -INFINITY;
  const size_t inner = walk_inner(&w), rows = walk_rows(&w);
  const ptrdiff_t to = w.stride[0][w.rank - 1];
  for (size_t r = 0; r < rows; ++r, walk_next_row(&w)) {
    const float *x = in + w.offset[1]; /* the input row is contiguous */
    float *y = out + w.offset[0];
    if (to == 0) {
      float max = *y;
      for (size_t i = 0; i < inner; ++i)
        max = x[i] > max ? x[i] : max;
      *y = max;
    } else {
      for (size_t i = 0; i < inner; ++i) {
        float *slot = y + (ptrdiff_t)i * to;
        *slot = x[i] > *slot ? x[i] : *slot;
      }
    }
  }
}

static void run_range(const NmIrNode *node, uint8_t dtype, void *out,
                      size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (dtype == T_F32)
      ((float *)out)[i] =
          (float)(node->cfg.range.start + (double)i * node->cfg.range.delta);
    else
      ((int64_t *)out)[i] = (int64_t)node->cfg.range.start +
                            (int64_t)i * (int64_t)node->cfg.range.delta;
  }
}

/* ---- Driver ------------------------------------------------------------- */

/* Buffers of finished values, reused by later values and later runs. */
typedef struct {
  void **data;
  size_t *capacity;
  size_t count;
  size_t cap;
} Pool;

static void pool_clear(Pool *pool) {
  for (size_t i = 0; i < pool->count; ++i)
    free(pool->data[i]);
  free(pool->data);
  free(pool->capacity);
  memset(pool, 0, sizeof(*pool));
}

/* The smallest pooled buffer of at least `bytes`, or a new one. */
static void *pool_take(Pool *pool, size_t bytes, size_t *capacity) {
  size_t best = SIZE_MAX;
  for (size_t i = 0; i < pool->count; ++i) {
    if (pool->capacity[i] >= bytes &&
        (best == SIZE_MAX || pool->capacity[i] < pool->capacity[best]))
      best = i;
  }
  if (best != SIZE_MAX) {
    void *data = pool->data[best];
    *capacity = pool->capacity[best];
    pool->count--;
    pool->data[best] = pool->data[pool->count];
    pool->capacity[best] = pool->capacity[pool->count];
    return data;
  }
  *capacity = bytes ? bytes : 1;
  return malloc(*capacity);
}

static bool pool_put(Pool *pool, void *data, size_t capacity) {
  if (pool->count == pool->cap) {
    size_t cap = pool->cap ? pool->cap * 2 : 32;
    void **grown = realloc(pool->data, cap * sizeof(*grown));
    if (!grown)
      return false;
    pool->data = grown;
    size_t *sizes = realloc(pool->capacity, cap * sizeof(*sizes));
    if (!sizes)
      return false;
    pool->capacity = sizes;
    pool->cap = cap;
  }
  pool->data[pool->count] = data;
  pool->capacity[pool->count] = capacity;
  pool->count++;
  return true;
}

/* State of one evaluation: per value its shape, type, data and how many
 * readers are still to come. */
typedef struct {
  const NmIrGraph *graph;
  size_t value_count; /* of the graph; kept for cleanup after it is gone */
  NmIrShape *shapes;
  uint8_t *dtypes;
  void **data;
  size_t *capacity;
  bool *owned;
  bool *ready;
  size_t *uses;
  /* Per node and input, the value a MatMul reads instead of its declared
   * input (NM_IR_INVALID_ID: the declared one); NULL for none. */
  const uint32_t *operand;
  Pool *pool; /* NULL: plain malloc/free */
} Run;

/* The graph inputs; graphs built before input lists existed have one. */
static size_t graph_inputs(const NmIrGraph *graph, uint32_t ids[NM_IR_MAX_INPUTS]) {
  if (graph->input_count == 0) {
    ids[0] = graph->model_input_id;
    return 1;
  }
  memcpy(ids, graph->input_ids, graph->input_count * sizeof(uint32_t));
  return graph->input_count;
}

static void release(Run *run, uint32_t id) {
  if (!run->owned[id])
    return;
  if (!run->pool || !pool_put(run->pool, run->data[id], run->capacity[id]))
    free(run->data[id]);
  run->data[id] = NULL;
  run->owned[id] = false;
}

static void run_release_all(Run *run) {
  if (run->data && run->owned) {
    for (size_t i = 0; i < run->value_count; ++i)
      release(run, (uint32_t)i);
  }
}

static void run_free(Run *run) {
  run_release_all(run);
  free(run->shapes);
  free(run->dtypes);
  free(run->data);
  free(run->capacity);
  free(run->owned);
  free(run->ready);
  free(run->uses);
  memset(run, 0, sizeof(*run));
}

/* Allocates the arrays and marks the constants ready (borrowed data). */
static NmStatus run_init(Run *run, const NmIrGraph *graph) {
  memset(run, 0, sizeof(*run));
  run->graph = graph;
  run->value_count = graph->value_count;
  size_t n = graph->value_count ? graph->value_count : 1;
  run->shapes = calloc(n, sizeof(*run->shapes));
  run->dtypes = calloc(n, sizeof(*run->dtypes));
  run->data = calloc(n, sizeof(*run->data));
  run->capacity = calloc(n, sizeof(*run->capacity));
  run->owned = calloc(n, sizeof(*run->owned));
  run->ready = calloc(n, sizeof(*run->ready));
  run->uses = calloc(n, sizeof(*run->uses));
  if (!run->shapes || !run->dtypes || !run->data || !run->capacity ||
      !run->owned || !run->ready || !run->uses) {
    run_free(run);
    return NM_STATUS_ALLOC_FAILED;
  }
  for (size_t i = 0; i < graph->value_count; ++i) {
    const NmIrValue *value = &graph->values[i];
    run->dtypes[i] = runtime_dtype(value->dtype);
    if (value->kind != NM_IR_VALUE_CONSTANT)
      continue;
    run->shapes[i] = value->shape;
    run->data[i] = value->raw ? value->raw : (void *)value->data;
    run->ready[i] = true;
  }
  return NM_STATUS_OK;
}

static bool node_is_source(const NmIrNode *node) {
  return node->op_type == NM_IR_OP_INPUT || node->op_type == NM_IR_OP_CONSTANT;
}

static bool is_view_op(uint8_t op) {
  return op == NM_IR_OP_RESHAPE || op == NM_IR_OP_SQUEEZE ||
         op == NM_IR_OP_UNSQUEEZE;
}

/* The value node `index` reads as input `j`. */
static uint32_t node_input(const Run *run, size_t index, uint32_t j) {
  const NmIrNode *node = &run->graph->nodes[index];
  if (run->operand && j < 2 && run->operand[2 * index + j] != NM_IR_INVALID_ID)
    return run->operand[2 * index + j];
  return node->input_ids[j];
}

/* Infers the shape of node `index` and, with `compute`, evaluates it. With
 * `release`, inputs whose last reader this is are released. */
static NmStatus run_node(Run *run, size_t index, bool compute, bool release_inputs) {
  const NmIrNode *node = &run->graph->nodes[index];
  NmIrShape in[NM_IR_MAX_NODE_INPUTS];
  uint8_t in_t[NM_IR_MAX_NODE_INPUTS] = {0, 0, 0};
  const void *src[NM_IR_MAX_NODE_INPUTS] = {NULL, NULL, NULL};
  uint32_t ids[NM_IR_MAX_NODE_INPUTS] = {0, 0, 0};
  for (uint32_t j = 0; j < node->input_count; ++j) {
    ids[j] = node_input(run, index, j);
    in[j] = run->shapes[ids[j]];
    in_t[j] = run->dtypes[ids[j]];
    src[j] = run->data[ids[j]];
  }
  NmIrShape shape;
  uint8_t dtype;
  if (!infer_shape(run->graph, node, in, in_t, &shape, &dtype))
    return NM_STATUS_UNSUPPORTED_TYPE;
  const uint32_t id = node->output_id;
  run->shapes[id] = shape;
  run->dtypes[id] = dtype;
  run->ready[id] = true;
  if (!compute)
    return NM_STATUS_OK;

  const size_t count = shape_numel(&shape);
  const size_t elem = elem_size(dtype);
  if (run->pool && is_view_op(node->op_type)) {
    /* Same elements in a new shape: take over the input's buffer when this
     * is its last reader, borrow constants and inputs, copy otherwise. */
    const uint32_t from = ids[0];
    if (!run->owned[from]) {
      run->data[id] = run->data[from];
      run->owned[id] = false;
    } else if (run->uses[from] == 1) {
      run->data[id] = run->data[from];
      run->capacity[id] = run->capacity[from];
      run->owned[id] = true;
      run->owned[from] = false;
      run->data[from] = NULL;
    } else {
      void *copy = pool_take(run->pool, count * elem, &run->capacity[id]);
      if (!copy)
        return NM_STATUS_ALLOC_FAILED;
      memcpy(copy, src[0], count * elem);
      run->data[id] = copy;
      run->owned[id] = true;
    }
    run->uses[from]--;
    return NM_STATUS_OK;
  }
  void *out = run->pool ? pool_take(run->pool, count * elem, &run->capacity[id])
                        : malloc((count ? count : 1) * elem);
  if (!out)
    return NM_STATUS_ALLOC_FAILED;
  if (!run->pool)
    run->capacity[id] = (count ? count : 1) * elem;
  run->data[id] = out;
  run->owned[id] = true;
  const uint8_t op = node->op_type;
  const float *f[NM_IR_MAX_NODE_INPUTS] = {src[0], src[1], src[2]};
  if (is_unary(op)) {
    run_unary(node, f[0], out, &shape);
  } else if (is_binary(op)) {
    run_binary(op, f[0], &in[0], f[1], &in[1], out, &shape);
  } else {
    switch (op) {
    case NM_IR_OP_MATMUL:
      if (!run_matmul(f[0], &in[0], f[1], &in[1], out, &shape))
        return NM_STATUS_INTERNAL_ERROR;
      break;
    case NM_IR_OP_TRANSPOSE:
      run_transpose(node, src[0], &in[0], elem, out, &shape);
      break;
    case NM_IR_OP_RESHAPE:
    case NM_IR_OP_SQUEEZE:
    case NM_IR_OP_UNSQUEEZE:
      memcpy(out, src[0], count * elem);
      break;
    case NM_IR_OP_LAYERNORM:
      run_layernorm(node, f[0], &in[0], node->input_count > 1 ? f[1] : NULL,
                    node->input_count > 2 ? f[2] : NULL, out);
      break;
    case NM_IR_OP_CONV2D:
      run_conv2d(node, f[0], &in[0], f[1], &in[1],
                 node->input_count > 2 ? f[2] : NULL, out, &shape);
      break;
    case NM_IR_OP_MAXPOOL2D:
      run_maxpool2d(node, f[0], &in[0], out, &shape);
      break;
    case NM_IR_OP_GATHER:
      if (!run_gather(node, src[0], &in[0], elem, src[1], shape_numel(&in[1]),
                      out))
        return NM_STATUS_INVALID_ARGUMENT; /* index out of range */
      break;
    case NM_IR_OP_EQUAL:
      run_equal(in_t[0], src[0], &in[0], src[1], &in[1], out, &shape);
      break;
    case NM_IR_OP_NOT:
      for (size_t i = 0; i < count; ++i)
        ((uint8_t *)out)[i] = ((const uint8_t *)src[0])[i] == 0;
      break;
    case NM_IR_OP_WHERE:
      run_where(src[0], &in[0], src[1], &in[1], src[2], &in[2], elem, out,
                &shape);
      break;
    case NM_IR_OP_CAST:
      run_cast(in_t[0], src[0], dtype, out, count);
      break;
    case NM_IR_OP_EXPAND: {
      Walk w;
      walk_init(&w, &shape, 1);
      broadcast_strides(&in[0], w.d, w.stride[0]);
      walk_copy(&w, src[0], elem, out);
      break;
    }
    case NM_IR_OP_SLICE:
      run_slice(node, src[0], &in[0], elem, out, &shape);
      break;
    case NM_IR_OP_REDUCEMAX:
      run_reducemax(node, f[0], &in[0], out, &shape);
      break;
    case NM_IR_OP_RANGE:
      run_range(node, dtype, out, count);
      break;
    default:
      return NM_STATUS_UNSUPPORTED_TYPE;
    }
  }
  /* Release inputs nobody reads any more. */
  for (uint32_t j = 0; release_inputs && j < node->input_count; ++j) {
    if (--run->uses[ids[j]] == 0)
      release(run, ids[j]);
  }
  return NM_STATUS_OK;
}

static bool interpretable(const NmIrGraph *graph) {
  for (size_t i = 0; i < graph->node_count; ++i) {
    const NmIrNode *node = &graph->nodes[i];
    if (node->input_count > NM_IR_MAX_NODE_INPUTS)
      return false;
    for (uint32_t j = 0; j < node->input_count; ++j) {
      if (node->input_ids[j] >= graph->value_count)
        return false;
    }
    if (node->output_id >= graph->value_count)
      return false;
  }
  for (size_t i = 0; i < graph->value_count; ++i) {
    const NmIrValue *value = &graph->values[i];
    if (value->kind == NM_IR_VALUE_CONSTANT &&
        ((!value->data && !value->raw) ||
         value->data_count != shape_numel(&value->shape)))
      return false;
  }
  return true;
}

static NmDType api_dtype(uint8_t dtype) {
  return dtype == T_I64 ? NM_DTYPE_INT64 : dtype == T_BOOL ? NM_DTYPE_BOOL
                                                           : NM_DTYPE_FLOAT32;
}

/* ---- Prepared graphs ---------------------------------------------------- */

struct NmInterp {
  const NmIrGraph *graph;
  size_t value_count;
  uint32_t inputs[NM_IR_MAX_INPUTS];
  size_t input_count;
  /* Nodes to evaluate per run, in order: live ones that do not depend on
   * constants alone (those are folded once, here). */
  uint32_t *order;
  size_t order_count;
  uint32_t *operand; /* see Run.operand */
  /* Values computed from constants alone, owned here. */
  void **folded;
  NmIrShape *folded_shape;
  uint8_t *folded_dtype;
  size_t *uses; /* readers per value and run */
  Run run;      /* reused by every run */
  Pool pool;
};

void graph_interp_free(NmInterp *interp) {
  if (!interp)
    return;
  run_free(&interp->run);
  pool_clear(&interp->pool);
  for (size_t i = 0; interp->folded && i < interp->value_count; ++i)
    free(interp->folded[i]);
  free(interp->folded);
  free(interp->folded_shape);
  free(interp->folded_dtype);
  free(interp->order);
  free(interp->operand);
  free(interp->uses);
  free(interp);
}

/* Runs the nodes in `order` (all non-source nodes in document order when NULL)
 * until each has run; GraphIR documents are topologically ordered, so this
 * is usually one pass. Shapes only unless `compute`. */
static NmStatus run_nodes(Run *run, const uint32_t *order, size_t count,
                          bool compute) {
  const NmIrGraph *graph = run->graph;
  bool *done = calloc(graph->node_count ? graph->node_count : 1, sizeof(bool));
  if (!done)
    return NM_STATUS_ALLOC_FAILED;
  size_t remaining = 0;
  for (size_t i = 0; i < graph->node_count; ++i) {
    done[i] = node_is_source(&graph->nodes[i]);
    remaining += !done[i];
  }
  if (order) {
    for (size_t i = 0; i < graph->node_count; ++i)
      done[i] = true;
    for (size_t i = 0; i < count; ++i)
      done[order[i]] = false;
    remaining = count;
  }
  NmStatus status = NM_STATUS_OK;
  while (remaining && status == NM_STATUS_OK) {
    size_t progress = 0;
    for (size_t i = 0; i < graph->node_count && status == NM_STATUS_OK; ++i) {
      if (done[i])
        continue;
      bool ready = true;
      for (uint32_t j = 0; j < graph->nodes[i].input_count; ++j)
        ready = ready && run->ready[node_input(run, i, j)];
      if (!ready)
        continue;
      status = run_node(run, i, compute, false);
      done[i] = true;
      progress++;
      remaining--;
    }
    if (!progress)
      status = NM_STATUS_FORMAT_ERROR; /* cycle or unreachable node */
  }
  free(done);
  if (status == NM_STATUS_OK && !order && !run->ready[graph->model_output_id])
    status = NM_STATUS_FORMAT_ERROR;
  return status;
}

/* Shape-only run with the open batch dimensions set to `batch` (0: the
 * declared batch); fills `shapes` (value_count entries) on success. */
static NmStatus dry_run(const NmIrGraph *graph, const uint32_t *ids,
                        size_t inputs, size_t batch, size_t *input_cols,
                        size_t *output_cols, NmIrShape *shapes) {
  Run run;
  NmStatus status = run_init(&run, graph);
  if (status != NM_STATUS_OK)
    return status;
  bool any_batch = false;
  for (size_t i = 0; i < inputs; ++i) {
    const NmIrValue *value = &graph->values[ids[i]];
    NmIrShape shape = value->shape;
    if (shape.ndim > 0 && shape.dims[0] == 0) {
      shape.dims[0] = batch ? batch : value->declared_batch;
      any_batch = true;
    }
    run.shapes[ids[i]] = shape;
    run.ready[ids[i]] = true;
  }
  status = run_nodes(&run, NULL, 0, false);
  if (status == NM_STATUS_OK) {
    const NmIrShape *output = &run.shapes[graph->model_output_id];
    const NmIrShape *first = &run.shapes[ids[0]];
    if (any_batch && (output->ndim < 1 || output->dims[0] != first->dims[0]))
      status = NM_STATUS_UNSUPPORTED_TYPE; /* output must be batch-major */
    else {
      size_t rows = output->ndim ? output->dims[0] : 1;
      size_t first_rows = first->ndim ? first->dims[0] : 1;
      *input_cols = first_rows ? shape_numel(first) / first_rows : 0;
      *output_cols = rows ? shape_numel(output) / rows : 0;
      memcpy(shapes, run.shapes, graph->value_count * sizeof(*shapes));
    }
  }
  run_free(&run);
  return status;
}

/* True for values computed from constants alone. */
static void mark_constant_derived(const NmIrGraph *graph, bool *derived) {
  for (size_t i = 0; i < graph->value_count; ++i)
    derived[i] = graph->values[i].kind == NM_IR_VALUE_CONSTANT;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < graph->node_count; ++i) {
      const NmIrNode *node = &graph->nodes[i];
      if (node_is_source(node) || derived[node->output_id])
        continue;
      bool all = true;
      for (uint32_t j = 0; j < node->input_count; ++j)
        all = all && derived[node->input_ids[j]];
      if (all) {
        derived[node->output_id] = true;
        changed = true;
      }
    }
  }
}

static const NmIrNode *producer(const NmIrGraph *graph, uint32_t value) {
  for (size_t i = 0; i < graph->node_count; ++i) {
    if (graph->nodes[i].output_id == value)
      return &graph->nodes[i];
  }
  return NULL;
}

static bool same_shape(const NmIrShape *a, const NmIrShape *b) {
  if (a->ndim != b->ndim)
    return false;
  for (size_t i = 0; i < a->ndim; ++i) {
    if (a->dims[i] != b->dims[i])
      return false;
  }
  return true;
}

/* MatMul broadcasts batch dimensions itself, so it can read a constant
 * operand before an Expand (and shape-preserving Reshapes) that only add
 * batch dimensions: PyTorch exports projections as MatMul(Expand(weight)),
 * which would otherwise copy the weight once per batch entry. */
static void elide_expands(const NmIrGraph *graph, const NmIrShape *shapes,
                          const bool *derived, uint32_t *operand) {
  for (size_t i = 0; i < graph->node_count; ++i) {
    operand[2 * i] = operand[2 * i + 1] = NM_IR_INVALID_ID;
    const NmIrNode *node = &graph->nodes[i];
    if (node->op_type != NM_IR_OP_MATMUL || node->input_count != 2)
      continue;
    for (uint32_t k = 0; k < 2; ++k) {
      uint32_t value = node->input_ids[k];
      const NmIrNode *from = producer(graph, value);
      while (from && from->op_type == NM_IR_OP_RESHAPE &&
             same_shape(&shapes[from->input_ids[0]], &shapes[value])) {
        value = from->input_ids[0];
        from = producer(graph, value);
      }
      if (!from || from->op_type != NM_IR_OP_EXPAND)
        continue;
      const uint32_t source = from->input_ids[0];
      const NmIrShape *s = &shapes[source], *e = &shapes[value],
                      *other = &shapes[node->input_ids[1 - k]],
                      *out = &shapes[node->output_id];
      if (!derived[source] || s->ndim < 2 ||
          s->dims[s->ndim - 1] != e->dims[e->ndim - 1] ||
          s->dims[s->ndim - 2] != e->dims[e->ndim - 2])
        continue;
      NmIrShape a = *other, b = *s, batch;
      a.ndim -= 2;
      b.ndim -= 2;
      NmIrShape want = *out;
      want.ndim -= 2;
      if (broadcast_shape(&a, &b, &batch) && same_shape(&batch, &want))
        operand[2 * i + k] = source;
    }
  }
}

NmStatus graph_interp_prepare(const NmIrGraph *graph, NmInterp **out,
                              size_t *input_cols, size_t *output_cols,
                              size_t *fixed_batch) {
  if (!graph || !out || graph->model_output_id >= graph->value_count ||
      !interpretable(graph))
    return NM_STATUS_INVALID_ARGUMENT;
  *out = NULL;
  NmInterp *interp = calloc(1, sizeof(*interp));
  if (!interp)
    return NM_STATUS_ALLOC_FAILED;
  interp->graph = graph;
  interp->value_count = graph->value_count;
  interp->input_count = graph_inputs(graph, interp->inputs);
  for (size_t i = 0; i < interp->input_count; ++i) {
    if (interp->inputs[i] >= graph->value_count) {
      graph_interp_free(interp);
      return NM_STATUS_INVALID_ARGUMENT;
    }
  }
  const size_t values = graph->value_count ? graph->value_count : 1;
  const size_t nodes = graph->node_count ? graph->node_count : 1;
  NmIrShape *shapes = calloc(values, sizeof(*shapes));
  bool *derived = calloc(values, sizeof(bool));
  bool *needed = calloc(values, sizeof(bool));
  bool *live = calloc(nodes, sizeof(bool));
  interp->operand = malloc(2 * nodes * sizeof(uint32_t));
  interp->folded = calloc(values, sizeof(void *));
  interp->folded_shape = calloc(values, sizeof(NmIrShape));
  interp->folded_dtype = calloc(values, sizeof(uint8_t));
  interp->uses = calloc(values, sizeof(size_t));
  interp->order = calloc(nodes, sizeof(uint32_t));
  NmStatus status = NM_STATUS_OK;
  if (!shapes || !derived || !needed || !live || !interp->operand ||
      !interp->folded || !interp->folded_shape || !interp->folded_dtype ||
      !interp->uses || !interp->order)
    status = NM_STATUS_ALLOC_FAILED;

  /* A dry run checks every operation and its shapes; inputs that take any
   * batch size get a batch of 1. Graphs that only work at their declared
   * batch (constants shaped by it) run at that size. */
  if (status == NM_STATUS_OK) {
    *fixed_batch = 0;
    status = dry_run(graph, interp->inputs, interp->input_count, 1, input_cols,
                     output_cols, shapes);
    const size_t declared = graph->values[interp->inputs[0]].declared_batch;
    if (status == NM_STATUS_UNSUPPORTED_TYPE && declared > 1 &&
        dry_run(graph, interp->inputs, interp->input_count, 0, input_cols,
                output_cols, shapes) == NM_STATUS_OK) {
      *fixed_batch = declared;
      status = NM_STATUS_OK;
    }
  }

  if (status == NM_STATUS_OK) {
    mark_constant_derived(graph, derived);
    elide_expands(graph, shapes, derived, interp->operand);
    /* Live nodes: those the output depends on, through the operands MatMul
     * really reads. Nodes come in topological order in GraphIR documents;
     * the loop repeats until nothing changes for others. */
    needed[graph->model_output_id] = true;
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t i = graph->node_count; i-- > 0;) {
        const NmIrNode *node = &graph->nodes[i];
        if (live[i] || !needed[node->output_id])
          continue;
        live[i] = true;
        changed = true;
        for (uint32_t j = 0; j < node->input_count; ++j) {
          uint32_t id = j < 2 && interp->operand[2 * i + j] != NM_IR_INVALID_ID
                            ? interp->operand[2 * i + j]
                            : node->input_ids[j];
          needed[id] = true;
        }
      }
    }
  }

  /* Fold live nodes on constants alone, once. */
  if (status == NM_STATUS_OK) {
    Run fold;
    status = run_init(&fold, graph);
    if (status == NM_STATUS_OK) {
      uint32_t *pending = calloc(nodes, sizeof(uint32_t));
      size_t count = 0;
      for (size_t i = 0; pending && i < graph->node_count; ++i) {
        if (live[i] && !node_is_source(&graph->nodes[i]) &&
            derived[graph->nodes[i].output_id])
          pending[count++] = (uint32_t)i;
      }
      if (!pending)
        status = NM_STATUS_ALLOC_FAILED;
      else if (count)
        status = run_nodes(&fold, pending, count, true);
      free(pending);
      for (size_t i = 0; status == NM_STATUS_OK && i < graph->value_count; ++i) {
        if (fold.owned[i] && needed[i]) {
          interp->folded[i] = fold.data[i];
          interp->folded_shape[i] = fold.shapes[i];
          interp->folded_dtype[i] = fold.dtypes[i];
          fold.owned[i] = false; /* now owned by interp */
        }
      }
      run_free(&fold);
    }
  }

  /* The nodes each run evaluates, and the readers of every value. */
  if (status == NM_STATUS_OK) {
    for (size_t i = 0; i < graph->node_count; ++i) {
      const NmIrNode *node = &graph->nodes[i];
      if (!live[i] || node_is_source(node) || derived[node->output_id])
        continue;
      interp->order[interp->order_count++] = (uint32_t)i;
      for (uint32_t j = 0; j < node->input_count; ++j) {
        uint32_t id = j < 2 && interp->operand[2 * i + j] != NM_IR_INVALID_ID
                          ? interp->operand[2 * i + j]
                          : node->input_ids[j];
        interp->uses[id]++;
      }
    }
    interp->uses[graph->model_output_id]++;
    status = run_init(&interp->run, graph);
    interp->run.operand = interp->operand;
    interp->run.pool = &interp->pool;
  }
  free(shapes);
  free(derived);
  free(needed);
  free(live);
  if (status != NM_STATUS_OK) {
    graph_interp_free(interp);
    return status;
  }
  *out = interp;
  return NM_STATUS_OK;
}

NmStatus graph_interp_bind(const NmIrGraph *graph, const NmInput *inputs,
                           size_t count, NmInput *ordered) {
  if (!graph || (!inputs && count))
    return NM_STATUS_INVALID_ARGUMENT;
  uint32_t ids[NM_IR_MAX_INPUTS];
  size_t expected = graph_inputs(graph, ids);
  if (count != expected)
    return NM_STATUS_INVALID_ARGUMENT;
  bool taken[NM_IR_MAX_INPUTS] = {false};
  for (size_t i = 0; i < count; ++i) {
    size_t slot = i;
    if (inputs[i].name) {
      slot = expected;
      for (size_t j = 0; j < expected; ++j) {
        if (ids[j] < graph->value_count &&
            !strcmp(graph->values[ids[j]].name, inputs[i].name))
          slot = j;
      }
    }
    if (slot >= expected || taken[slot] || ids[slot] >= graph->value_count)
      return NM_STATUS_INVALID_ARGUMENT;
    taken[slot] = true;
    const NmIrValue *value = &graph->values[ids[slot]];
    const NmInput *input = &inputs[i];
    if (!input->data || input->dtype != api_dtype(runtime_dtype(value->dtype)) ||
        input->ndim != value->shape.ndim)
      return NM_STATUS_INVALID_ARGUMENT;
    for (size_t d = 0; d < input->ndim; ++d) {
      /* 0 marks a dimension of any size (the batch). */
      if (input->dims[d] == 0 ||
          (value->shape.dims[d] != 0 && input->dims[d] != value->shape.dims[d]))
        return NM_STATUS_INVALID_ARGUMENT;
    }
    ordered[slot] = *input;
  }
  /* Inputs of any batch size must agree on it. */
  size_t batch = 0;
  for (size_t i = 0; i < expected; ++i) {
    if (graph->values[ids[i]].shape.ndim > 0 &&
        graph->values[ids[i]].shape.dims[0] == 0) {
      if (batch && ordered[i].dims[0] != batch)
        return NM_STATUS_INVALID_ARGUMENT;
      batch = ordered[i].dims[0];
    }
  }
  return NM_STATUS_OK;
}

static void to_float(uint8_t dtype, const void *in, float *out, size_t n) {
  if (dtype == T_F32) {
    memcpy(out, in, n * sizeof(float));
    return;
  }
  for (size_t i = 0; i < n; ++i)
    out[i] = dtype == T_I64 ? (float)((const int64_t *)in)[i]
                            : (float)(((const uint8_t *)in)[i] != 0);
}

NmStatus graph_interp_run(NmInterp *interp, const NmInput *inputs,
                          FloatMatrix **out) {
  if (!interp || !inputs || !out)
    return NM_STATUS_INVALID_ARGUMENT;
  *out = NULL;
  const NmIrGraph *graph = interp->graph;
  Run *run = &interp->run;
  /* Reset: constants and folded values ready, everything else pending. */
  for (size_t i = 0; i < graph->value_count; ++i) {
    const NmIrValue *value = &graph->values[i];
    run->owned[i] = false;
    run->uses[i] = interp->uses[i];
    if (interp->folded[i]) {
      run->shapes[i] = interp->folded_shape[i];
      run->dtypes[i] = interp->folded_dtype[i];
      run->data[i] = interp->folded[i];
      run->ready[i] = true;
    } else if (value->kind == NM_IR_VALUE_CONSTANT) {
      run->shapes[i] = value->shape;
      run->dtypes[i] = runtime_dtype(value->dtype);
      run->data[i] = value->raw ? value->raw : (void *)value->data;
      run->ready[i] = true;
    } else {
      run->data[i] = NULL;
      run->ready[i] = false;
    }
  }
  for (size_t i = 0; i < interp->input_count; ++i) {
    const uint32_t id = interp->inputs[i];
    NmIrShape shape = {.ndim = inputs[i].ndim};
    memcpy(shape.dims, inputs[i].dims, shape.ndim * sizeof(size_t));
    run->shapes[id] = shape;
    run->dtypes[id] = runtime_dtype(graph->values[id].dtype);
    run->data[id] = (void *)inputs[i].data; /* borrowed */
    run->ready[id] = true;
  }
  NmStatus status = NM_STATUS_OK;
  for (size_t i = 0; i < interp->order_count && status == NM_STATUS_OK; ++i)
    status = run_node(run, interp->order[i], true, true);
  if (status == NM_STATUS_OK) {
    const uint32_t id = graph->model_output_id;
    const NmIrShape *shape = &run->shapes[id];
    size_t n = shape_numel(shape);
    size_t rows = shape->ndim ? shape->dims[0] : 1;
    *out = rows ? sm_create(rows, n / rows) : NULL;
    if (!*out)
      status = NM_STATUS_ALLOC_FAILED;
    else
      to_float(run->dtypes[id], run->data[id], (*out)->values, n);
  }
  run_release_all(run);
  return status;
}
