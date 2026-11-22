/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef MNIST_DATA_H
#define MNIST_DATA_H

#include "sm.h"

FloatMatrix *load_mnist_images(const char *filename, size_t max_images);
FloatMatrix *load_mnist_labels_onehot(const char *filename, size_t max_labels,
                                      size_t num_classes);

#endif // MNIST_DATA_H