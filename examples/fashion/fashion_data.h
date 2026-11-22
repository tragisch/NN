/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef FASHION_DATA_H
#define FASHION_DATA_H

#include "sm.h"

FloatMatrix *load_fashion_images(const char *filename, size_t max_images);
FloatMatrix *load_fashion_labels_onehot(const char *filename, size_t max_labels,
                                        size_t num_classes);

#endif // FASHION_DATA_H
