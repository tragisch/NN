/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* GraphIR v4 export for finalized frontend models and graphs. */
#ifndef NM_EXPORT_GRAPHIR_H
#define NM_EXPORT_GRAPHIR_H

#include "nm_frontend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  size_t batch_size;
  /* NULL writes the weights inline into the JSON. Otherwise they go into a
   * `.girw` weight package (version 2, checksummed) at this path and the JSON
   * refers to it by file name, so the two files belong in one directory. */
  const char *weights_path;
} NmGraphIrExportOptions;

NmStatus nm_export_graphir_json(const NmFrontendModel *model,
                                const NmGraphIrExportOptions *options,
                                const char *path);

/* Same for any graph, for example an imported or trained one. */
NmStatus nm_export_graphir_graph_json(const NmGraphModel *graph,
                                      const NmGraphIrExportOptions *options,
                                      const char *path);

#ifdef __cplusplus
}
#endif

#endif /* NM_EXPORT_GRAPHIR_H */
