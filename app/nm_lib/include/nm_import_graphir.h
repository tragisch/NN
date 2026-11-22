/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* GraphIR v3/v4 JSON import. The result is an executable graph that does not
 * depend on the frontend model: the engine runs it directly. */
#ifndef NM_IMPORT_GRAPHIR_H
#define NM_IMPORT_GRAPHIR_H

#include "nm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Reads a GraphIR document with inline float32 weights. Supported operations:
 * Input, Constant, MatMul + Add, ReLU, Sigmoid, Tanh, Softmax, Conv2D,
 * MaxPool2D and Reshape to [batch, features]. The graph is only checked for
 * well-formedness here; nm_engine_c_prepare_graph validates the structure.
 * The batch dimension of activations is treated as variable. */
NmStatus nm_import_graphir_json(const char *path, NmGraphModel **out_graph);

void nm_graph_model_free(NmGraphModel *graph);

#ifdef __cplusplus
}
#endif

#endif /* NM_IMPORT_GRAPHIR_H */
