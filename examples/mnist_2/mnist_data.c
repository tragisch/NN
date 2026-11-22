/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "mnist_data.h"
#include <limits.h>
#include <log.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *open_mnist_file(const char *filename, const char *mode) {
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

FloatMatrix *load_mnist_labels_onehot(const char *filename, size_t max_labels,
                                      size_t num_classes) {
  if (!filename || num_classes == 0)
    return NULL;

  FILE *file = open_mnist_file(filename, "rb");
  if (!file) {
    log_error("Failed to open label file: %s", filename);
    return NULL;
  }

  uint32_t magic_number = 0, num_labels = 0;
  if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1 ||
      fread(&num_labels, sizeof(uint32_t), 1, file) != 1) {
    log_error("Error reading MNIST label header from %s", filename);
    fclose(file);
    return NULL;
  }

  magic_number = __builtin_bswap32(magic_number);
  num_labels = __builtin_bswap32(num_labels);

  if (magic_number != 2049) {
    log_error("Invalid magic number in label file: %s", filename);
    fclose(file);
    return NULL;
  }

  if (max_labels > num_labels) {
    max_labels = num_labels;
  }

  FloatMatrix *labels = sm_create_zeros(max_labels, num_classes);
  if (!labels) {
    fclose(file);
    return NULL;
  }

  for (size_t i = 0; i < max_labels; ++i) {
    uint8_t label = 0;
    if (fread(&label, sizeof(uint8_t), 1, file) != 1) {
      log_error("Error reading label %zu from %s", i, filename);
      sm_destroy(labels);
      fclose(file);
      return NULL;
    }
    if ((size_t)label >= num_classes) {
      log_error("Invalid label value %u at index %zu in %s", (unsigned)label,
                i, filename);
      sm_destroy(labels);
      fclose(file);
      return NULL;
    }

    labels->values[i * num_classes + (size_t)label] = 1.0f;
  }

  if (fclose(file) != 0) {
    sm_destroy(labels);
    return NULL;
  }

  return labels;
}

FloatMatrix *load_mnist_images(const char *filename, size_t max_images) {
  if (!filename)
    return NULL;

  FILE *file = open_mnist_file(filename, "rb");
  if (!file) {
    log_error("Failed to open image file: %s", filename);
    return NULL;
  }

  uint32_t magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
  if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1 ||
      fread(&num_images, sizeof(uint32_t), 1, file) != 1 ||
      fread(&num_rows, sizeof(uint32_t), 1, file) != 1 ||
      fread(&num_cols, sizeof(uint32_t), 1, file) != 1) {
    log_error("Error reading MNIST image header from %s", filename);
    fclose(file);
    return NULL;
  }

  // Convert from big-endian to little-endian
  magic_number = __builtin_bswap32(magic_number);
  num_images = __builtin_bswap32(num_images);
  num_rows = __builtin_bswap32(num_rows);
  num_cols = __builtin_bswap32(num_cols);

  if (magic_number != 2051) {
    log_error("Error reading MNIST image header from %s", filename);
    fclose(file);
    return NULL;
  }

  if (max_images > num_images) {
    max_images = num_images;
  }

  if (num_rows == 0 || num_cols == 0 ||
      (size_t)num_rows > SIZE_MAX / (size_t)num_cols) {
    fclose(file);
    return NULL;
  }

  size_t image_size = (size_t)num_rows * (size_t)num_cols;

  if (max_images > 0 && image_size > SIZE_MAX / max_images) {
    fclose(file);
    return NULL;
  }

  FloatMatrix *mat = sm_create(max_images, image_size);
  if (!mat) {
    fclose(file);
    return NULL;
  }

  uint8_t *buffer = (uint8_t *)malloc(image_size);
  if (!buffer) {
    sm_destroy(mat);
    fclose(file);
    return NULL;
  }

  for (size_t i = 0; i < max_images; ++i) {
    if (fread(buffer, sizeof(uint8_t), image_size, file) != image_size) {
      log_error("Error reading image %zu from %s", i, filename);
      free(buffer);
      sm_destroy(mat);
      fclose(file);
      return NULL;
    }
    for (size_t j = 0; j < image_size; ++j) {
      mat->values[i * image_size + j] = buffer[j] / 255.0f;
    }
  }

  free(buffer);

  if (fclose(file) != 0) {
    sm_destroy(mat);
    return NULL;
  }

  return mat;
}

