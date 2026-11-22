/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef IRIS_DATA_H
#define IRIS_DATA_H

#include "sm.h"

// Load Iris dataset from CSV file
// Returns: X (features) and Y (one-hot encoded labels)
// Caller must destroy returned matrices
int load_iris_dataset(const char *filepath, 
                      FloatMatrix **out_X, 
                      FloatMatrix **out_Y,
                      size_t *out_num_samples);

#endif // IRIS_DATA_H
