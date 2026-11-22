/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Typed, named model inputs (token ids, masks) for nm_api.h and the C engine. */
#ifndef NM_INPUT_H
#define NM_INPUT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Element type of a model input. */
typedef enum {
  NM_DTYPE_FLOAT32 = 0,
  NM_DTYPE_INT64 = 1,
  NM_DTYPE_BOOL = 2, /* one byte per element, 0 or 1 */
} NmDType;

#define NM_INPUT_MAX_DIMS 6

/* One named model input, borrowed for the duration of a call: row-major
 * elements of `dtype` (float, int64_t or one byte per bool). `name` is the
 * GraphIR input name; NULL matches the input at the same position. */
typedef struct {
  const char *name;
  NmDType dtype;
  const void *data;
  size_t ndim; /* 0..NM_INPUT_MAX_DIMS */
  size_t dims[NM_INPUT_MAX_DIMS];
} NmInput;

#ifdef __cplusplus
}
#endif

#endif /* NM_INPUT_H */
