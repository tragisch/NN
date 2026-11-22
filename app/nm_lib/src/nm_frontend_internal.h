/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Internal frontend helpers shared between nm_lib modules. */
#ifndef NM_FRONTEND_INTERNAL_H
#define NM_FRONTEND_INTERNAL_H

#include "nm_frontend.h"
#include "nm_model_types_internal.h"

/* Turns a loaded legacy model into a finalized frontend model. On success the
 * frontend model owns the weights (copied) and `seq` is released; on failure
 * the caller still owns `seq`. */
NmStatus nm_frontend_adopt_seq(NmSeq *seq, NmFrontendModel **out_model);

#endif /* NM_FRONTEND_INTERNAL_H */
