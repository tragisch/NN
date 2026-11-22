/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef MOONS_DATA_H
#define MOONS_DATA_H

#include "sm.h"

// Load the make_moons dataset from a CSV file (columns: x1,x2,label).
// Returns: X (features, N x 2) and Y (one-hot encoded labels, N x 2).
// Caller must destroy the returned matrices.
int load_moons_dataset(const char *filepath,
                       FloatMatrix **out_X,
                       FloatMatrix **out_Y,
                       size_t *out_num_samples);

#endif // MOONS_DATA_H
