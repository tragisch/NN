/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "cifar_data.h"
#include <limits.h>
#include <log.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CIFAR10_IMAGE_SIZE 3072  // 32x32x3 (RGB)
#define CIFAR10_NUM_CLASSES 10
#define CIFAR10_RECORD_SIZE 3073 // 1 byte label + 3072 bytes pixels
#define CIFAR10_IMAGES_PER_BATCH 10000

static FILE *open_cifar_file(const char *filename, const char *mode) {
  if (!filename || !mode)
    return NULL;

  FILE *file = fopen(filename, mode);
  if (file)
    return file;

  char path[PATH_MAX];
  const char *runfiles_dir = getenv("RUNFILES_DIR");
  if (runfiles_dir &&
      snprintf(path, sizeof(path), "%s/_main/%s", runfiles_dir, filename) > 0 &&
      strlen(path) < sizeof(path)) {
    file = fopen(path, mode);
    if (file)
      return file;
  }

  const char *test_srcdir = getenv("TEST_SRCDIR");
  const char *test_workspace = getenv("TEST_WORKSPACE");
  if (test_srcdir && test_workspace &&
      snprintf(path, sizeof(path), "%s/%s/%s", test_srcdir, test_workspace,
               filename) > 0 &&
      strlen(path) < sizeof(path)) {
    file = fopen(path, mode);
    if (file)
      return file;
  }

  return NULL;
}

int load_cifar10_batch(const char *filepath, 
                       size_t max_images,
                       FloatMatrix **out_X, 
                       FloatMatrix **out_Y) {
  if (!filepath || !out_X || !out_Y)
    return -1;

  *out_X = NULL;
  *out_Y = NULL;

  FILE *file = open_cifar_file(filepath, "rb");
  if (!file) {
    log_error("Failed to open CIFAR-10 batch file: %s", filepath);
    return -1;
  }

  // Determine file size
  fseek(file, 0, SEEK_END);
  long file_size = ftell(file);
  fseek(file, 0, SEEK_SET);

  size_t num_images = file_size / CIFAR10_RECORD_SIZE;
  if (file_size % CIFAR10_RECORD_SIZE != 0) {
    log_error("Invalid CIFAR-10 file size: %ld (expected multiple of %d)",
              file_size, CIFAR10_RECORD_SIZE);
    fclose(file);
    return -1;
  }

  if (max_images == 0 || max_images > num_images) {
    max_images = num_images;
  }

  // Allocate matrices
  FloatMatrix *X = sm_create(max_images, CIFAR10_IMAGE_SIZE);
  FloatMatrix *Y = sm_create_zeros(max_images, CIFAR10_NUM_CLASSES);
  if (!X || !Y) {
    log_error("Failed to allocate memory for CIFAR-10 data");
    sm_destroy(X);
    sm_destroy(Y);
    fclose(file);
    return -1;
  }

  uint8_t record[CIFAR10_RECORD_SIZE];
  
  for (size_t i = 0; i < max_images; ++i) {
    if (fread(record, 1, CIFAR10_RECORD_SIZE, file) != CIFAR10_RECORD_SIZE) {
      log_error("Error reading image %zu from %s", i, filepath);
      sm_destroy(X);
      sm_destroy(Y);
      fclose(file);
      return -1;
    }

    // First byte is the label
    uint8_t label = record[0];
    if (label >= CIFAR10_NUM_CLASSES) {
      log_error("Invalid label %u at image %zu in %s", label, i, filepath);
      sm_destroy(X);
      sm_destroy(Y);
      fclose(file);
      return -1;
    }

    // One-hot encode label
    Y->values[i * CIFAR10_NUM_CLASSES + label] = 1.0f;

    // Next 3072 bytes are pixels (R, G, B channels)
    // Normalize to [0, 1]
    for (size_t j = 0; j < CIFAR10_IMAGE_SIZE; ++j) {
      X->values[i * CIFAR10_IMAGE_SIZE + j] = record[j + 1] / 255.0f;
    }
  }

  fclose(file);
  *out_X = X;
  *out_Y = Y;
  return 0;
}

int load_cifar10_train(const char *dir_prefix,
                       size_t max_images_per_batch,
                       FloatMatrix **out_X, 
                       FloatMatrix **out_Y) {
  if (!dir_prefix || !out_X || !out_Y)
    return -1;

  *out_X = NULL;
  *out_Y = NULL;

  // Load each of the 5 training batches
  FloatMatrix *batches_X[5] = {NULL};
  FloatMatrix *batches_Y[5] = {NULL};
  size_t total_images = 0;

  for (int batch = 1; batch <= 5; ++batch) {
    char filepath[PATH_MAX];
    snprintf(filepath, sizeof(filepath), "%s/data_batch_%d.bin", 
             dir_prefix, batch);

    if (load_cifar10_batch(filepath, max_images_per_batch,
                          &batches_X[batch-1], &batches_Y[batch-1]) != 0) {
      log_error("Failed to load batch %d", batch);
      // Cleanup
      for (int i = 0; i < 5; ++i) {
        sm_destroy(batches_X[i]);
        sm_destroy(batches_Y[i]);
      }
      return -1;
    }
    total_images += batches_X[batch-1]->rows;
  }

  // Concatenate all batches
  FloatMatrix *X = sm_create(total_images, CIFAR10_IMAGE_SIZE);
  FloatMatrix *Y = sm_create(total_images, CIFAR10_NUM_CLASSES);
  if (!X || !Y) {
    sm_destroy(X);
    sm_destroy(Y);
    for (int i = 0; i < 5; ++i) {
      sm_destroy(batches_X[i]);
      sm_destroy(batches_Y[i]);
    }
    return -1;
  }

  size_t offset = 0;
  for (int batch = 0; batch < 5; ++batch) {
    size_t batch_size = batches_X[batch]->rows;
    
    // Copy X
    memcpy(&X->values[offset * CIFAR10_IMAGE_SIZE],
           batches_X[batch]->values,
           batch_size * CIFAR10_IMAGE_SIZE * sizeof(float));
    
    // Copy Y
    memcpy(&Y->values[offset * CIFAR10_NUM_CLASSES],
           batches_Y[batch]->values,
           batch_size * CIFAR10_NUM_CLASSES * sizeof(float));
    
    offset += batch_size;
    
    // Free individual batch
    sm_destroy(batches_X[batch]);
    sm_destroy(batches_Y[batch]);
  }

  *out_X = X;
  *out_Y = Y;
  return 0;
}
