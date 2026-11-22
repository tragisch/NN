/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm_frontend.h"
#include "nm_frontend_internal.h"
#include "nm_model_internal.h"

NmStatus nm_frontend_import_legacy(const char *path,
                                   NmFrontendModel **out_model) {
  if (!path || !out_model)
    return NM_STATUS_INVALID_ARGUMENT;
  *out_model = NULL;
  NmSeq *seq = NULL;
  NmStatus status = nm_seq_load(path, &seq);
  if (status != NM_STATUS_OK)
    return status;
  status = nm_frontend_adopt_seq(seq, out_model);
  if (status != NM_STATUS_OK)
    nm_seq_free(seq);
  return status;
}
