/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef CIFAR_DATA_H
#define CIFAR_DATA_H

#include "sm.h"

// Load CIFAR-10 images and labels from binary batch file
// Format: Each record is 3073 bytes (1 byte label + 3072 bytes RGB pixel data)
// Returns: X (features normalized to [0,1]) and Y (one-hot encoded labels)
// max_images: maximum number of images to load (0 = load all 10000)
int load_cifar10_batch(const char *filepath, 
                       size_t max_images,
                       FloatMatrix **out_X, 
                       FloatMatrix **out_Y);

// Load all 5 training batches and concatenate them
int load_cifar10_train(const char *dir_prefix,
                       size_t max_images_per_batch,
                       FloatMatrix **out_X, 
                       FloatMatrix **out_Y);

#endif // CIFAR_DATA_H
