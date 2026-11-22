/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/*
 * Internal graph IR. It mirrors the GraphIR document model: named values with
 * static tensor contracts, constants that carry data, and nodes that name an
 * operation. Activation values use dims[0] == 0 for "any batch size".
 */

#ifndef NM_GRAPH_IR_INTERNAL_H
#define NM_GRAPH_IR_INTERNAL_H

#include "nm.h"
#include "st.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NM_IR_INVALID_ID UINT32_MAX
#define NM_IR_MAX_NODE_INPUTS 3U
#define NM_IR_MAX_DIMS 6U
#define NM_IR_MAX_INPUTS 8U

typedef enum {
  NM_IR_VALUE_MODEL_INPUT = 0,
  NM_IR_VALUE_CONSTANT = 1,
  NM_IR_VALUE_INTERMEDIATE = 2,
  NM_IR_VALUE_MODEL_OUTPUT = 3,
} NmIrValueKind;

/* Operation names follow GraphIR. */
typedef enum {
  NM_IR_OP_INPUT = 0,
  NM_IR_OP_CONSTANT = 1,
  NM_IR_OP_MATMUL = 2,
  NM_IR_OP_ADD = 3,
  NM_IR_OP_RELU = 4,
  NM_IR_OP_SIGMOID = 5,
  NM_IR_OP_TANH = 6,
  NM_IR_OP_SOFTMAX = 7,
  NM_IR_OP_CONV2D = 8,
  NM_IR_OP_MAXPOOL2D = 9,
  NM_IR_OP_RESHAPE = 10,
  NM_IR_OP_SUB = 11,
  NM_IR_OP_MUL = 12,
  NM_IR_OP_DIV = 13,
  NM_IR_OP_TRANSPOSE = 14,
  NM_IR_OP_LAYERNORM = 15,
  NM_IR_OP_GELU = 16,
  /* Tensor operations for token embeddings and attention masks; they run on
   * the interpreter only. */
  NM_IR_OP_GATHER = 17,
  NM_IR_OP_EQUAL = 18,
  NM_IR_OP_NOT = 19,
  NM_IR_OP_WHERE = 20,
  NM_IR_OP_CAST = 21,
  NM_IR_OP_EXPAND = 22,
  NM_IR_OP_SLICE = 23,
  NM_IR_OP_REDUCEMAX = 24,
  NM_IR_OP_RANGE = 25,
  NM_IR_OP_SQUEEZE = 26,
  NM_IR_OP_UNSQUEEZE = 27,
} NmIrOpType;

typedef struct {
  size_t ndim;
  size_t dims[NM_IR_MAX_DIMS];
} NmIrShape;

/* Element type of a value as GraphIR declares it. The engine computes every
 * float graph in f32 ("promoted"): 16-bit weights are widened on import, and
 * export narrows them again, so a 16-bit model keeps its type. Integer and
 * boolean values (token ids, masks) run on the interpreter: int64 elements are
 * int64_t, bool elements one byte (0 or 1). */
typedef enum {
  NM_IR_DTYPE_F32 = 0,
  NM_IR_DTYPE_BF16 = 1,
  NM_IR_DTYPE_F16 = 2,
  NM_IR_DTYPE_I64 = 3,
  NM_IR_DTYPE_BOOL = 4,
} NmIrDType;

static inline bool nm_ir_dtype_is_float(uint8_t dtype) {
  return dtype == NM_IR_DTYPE_F32 || dtype == NM_IR_DTYPE_BF16 ||
         dtype == NM_IR_DTYPE_F16;
}

const char *nm_ir_dtype_name(NmIrDType dtype);

typedef struct {
  char name[64];
  NmIrShape shape;
  uint8_t kind;
  uint8_t dtype; /* NmIrDType */
  /* First dimension as the document declares it, kept when the importer
   * opens the batch (shape.dims[0] == 0); 0 otherwise. */
  size_t declared_batch;
  uint32_t producer_node_id;
  /* Constants only. `data` points at the float32 elements. Rank 1 and rank 4
   * constants also carry a tensor view so that CNN kernels can use them. The
   * graph frees `data` / `tensor` only when `owns_data` is set; graphs lowered
   * from a model borrow the model's live weights. int64 and bool constants
   * keep their elements in `raw` instead (always owned) and have no `data`. */
  float *data;
  size_t data_count;
  FloatTensor *tensor;
  void *raw;
  bool owns_data;
} NmIrValue;

typedef struct {
  uint8_t op_type;
  uint32_t input_ids[NM_IR_MAX_NODE_INPUTS];
  uint32_t input_count;
  uint32_t output_id;
  union {
    struct {
      size_t stride_h;
      size_t stride_w;
      size_t pad_h;
      size_t pad_w;
      NmConvAlgo algo;
    } conv2d;
    struct {
      size_t kernel_h;
      size_t kernel_w;
      size_t stride_h;
      size_t stride_w;
      size_t pad_h;
      size_t pad_w;
    } maxpool2d;
    struct {
      size_t perm[NM_IR_MAX_DIMS];
    } transpose;
    struct {
      size_t norm_rank; /* trailing dimensions that are normalized */
      float epsilon;
    } layernorm;
    struct {
      bool exact; /* erf form; the default is the tanh approximation */
    } gelu;
    struct {
      size_t axis;
    } gather;
    struct {
      uint8_t to; /* NmIrDType */
    } cast;
    struct {
      size_t ndim;
      size_t dims[NM_IR_MAX_DIMS];
    } expand;
    struct {
      /* Per input axis; axes the node does not slice keep 0, INT64_MAX, 1. */
      int64_t starts[NM_IR_MAX_DIMS];
      int64_t ends[NM_IR_MAX_DIMS];
      int64_t steps[NM_IR_MAX_DIMS];
    } slice;
    struct {
      /* Bit i set: axis i is reduced (ReduceMax), removed (Squeeze) or
       * inserted at output position i (Unsqueeze). */
      uint32_t axes;
      bool keepdims;
    } axes;
    struct {
      double start;
      double delta;
    } range;
  } cfg;
} NmIrNode;

typedef struct NmIrGraph {
  NmIrValue *values;
  size_t value_count;
  size_t value_capacity;
  NmIrNode *nodes;
  size_t node_count;
  size_t node_capacity;
  /* The first input; graphs with one float input run on the execution plan
   * or take a batch of any size. */
  uint32_t model_input_id;
  /* Every input in document order (model_input_id is input_ids[0]). Graphs
   * with several inputs keep their declared shapes, batch included. */
  uint32_t input_ids[NM_IR_MAX_INPUTS];
  size_t input_count;
  uint32_t model_output_id;
  /* Set while the constants borrow memory from a weight set (nm_weightset.h);
   * the graph calls `binding_free` when it is reset or destroyed. */
  void *binding;
  void (*binding_free)(void *binding);
} NmIrGraph;

bool nm_ir_graph_init(NmIrGraph *graph);
void nm_ir_graph_reset(NmIrGraph *graph);
void nm_ir_graph_destroy(NmIrGraph *graph);

/* Append helpers. Both return NM_IR_INVALID_ID on allocation failure. */
uint32_t nm_ir_graph_add_value(NmIrGraph *graph, const char *name,
                               NmIrShape shape, NmIrValueKind kind);
uint32_t nm_ir_graph_add_node(NmIrGraph *graph, const NmIrNode *node);

/* Deep-copies `src` into the empty graph `dst`; every constant becomes owned by
 * `dst`. Used to detach a lowered graph from the model whose weights it borrows. */
bool nm_ir_graph_clone_owned(NmIrGraph *dst, const NmIrGraph *src);

const char *nm_ir_op_name(NmIrOpType op);
bool nm_ir_op_from_name(const char *name, NmIrOpType *out);

#ifdef __cplusplus
}
#endif

#endif /* NM_GRAPH_IR_INTERNAL_H */
