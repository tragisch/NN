/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Model-definition frontend. It collects the topology and, on finalize,
 * produces an executable graph that owns the initialized weights. */
#ifndef NM_FRONTEND_H
#define NM_FRONTEND_H

#include "nm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NmFrontendModel NmFrontendModel;

NmFrontendModel *nm_frontend_create(NmShape input_shape);
void nm_frontend_free(NmFrontendModel *model);

NmStatus nm_frontend_add_dense(NmFrontendModel *model,
                               const NmDenseConfig *config);
NmStatus nm_frontend_add_conv2d(NmFrontendModel *model,
                                const NmConv2dConfig *config);
NmStatus nm_frontend_add_maxpool2d(NmFrontendModel *model,
                                   const NmPool2dConfig *config);
NmStatus nm_frontend_add_flatten(NmFrontendModel *model);

/* Finalization validates the topology, initializes the weights and builds the
 * graph. It never prepares a runtime. */
NmStatus nm_frontend_finalize(NmFrontendModel *model);

/* The graph of a finalized model (NULL before finalize). The model owns it;
 * the engine runs and trains it in place. */
NmGraphModel *nm_frontend_graph(NmFrontendModel *model);

/* Imports topology and weights from a legacy .nms* file. Runtime state is
 * intentionally discarded and the returned model is already finalized. */
NmStatus nm_frontend_import_legacy(const char *path,
                                   NmFrontendModel **out_model);

#ifdef __cplusplus
}
#endif

#endif /* NM_FRONTEND_H */
