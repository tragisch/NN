/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "moons_data.h"
#include <limits.h>
#include <log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *open_moons_file(const char *filename) {
  if (!filename)
    return NULL;

  FILE *file = fopen(filename, "r");
  if (file)
    return file;

  // Try with Bazel runfiles
  char path[PATH_MAX];
  const char *runfiles_dir = getenv("RUNFILES_DIR");
  if (runfiles_dir &&
      snprintf(path, sizeof(path), "%s/_main/%s", runfiles_dir, filename) > 0 &&
      strlen(path) < sizeof(path)) {
    file = fopen(path, "r");
    if (file)
      return file;
  }

  const char *test_srcdir = getenv("TEST_SRCDIR");
  const char *test_workspace = getenv("TEST_WORKSPACE");
  if (test_srcdir && test_workspace &&
      snprintf(path, sizeof(path), "%s/%s/%s", test_srcdir, test_workspace,
               filename) > 0 &&
      strlen(path) < sizeof(path)) {
    file = fopen(path, "r");
    if (file)
      return file;
  }

  return NULL;
}

int load_moons_dataset(const char *filepath, FloatMatrix **out_X,
                       FloatMatrix **out_Y, size_t *out_num_samples) {
  if (!filepath || !out_X || !out_Y || !out_num_samples)
    return -1;

  FILE *file = open_moons_file(filepath);
  if (!file) {
    log_error("Failed to open moons data file: %s", filepath);
    return -1;
  }

  // First pass: count valid lines
  size_t num_samples = 0;
  char line[256];
  while (fgets(line, sizeof(line), file)) {
    if (strlen(line) <= 1)
      continue;
    num_samples++;
  }

  if (num_samples == 0) {
    log_error("No data found in file");
    fclose(file);
    return -1;
  }

  float *X_data = (float *)malloc(num_samples * 2 * sizeof(float));
  float *Y_data = (float *)malloc(num_samples * 2 * sizeof(float));
  if (!X_data || !Y_data) {
    log_error("Memory allocation failed");
    free(X_data);
    free(Y_data);
    fclose(file);
    return -1;
  }

  // Second pass: parse CSV: x1,x2,label
  rewind(file);
  size_t idx = 0;
  while (fgets(line, sizeof(line), file) && idx < num_samples) {
    if (strlen(line) <= 1)
      continue;

    float x1, x2;
    int label;
    if (sscanf(line, "%f,%f,%d", &x1, &x2, &label) != 3) {
      log_error("Failed to parse line %zu: %s", idx + 1, line);
      free(X_data);
      free(Y_data);
      fclose(file);
      return -1;
    }

    X_data[idx * 2] = x1;
    X_data[idx * 2 + 1] = x2;
    // One-hot: class 0 -> (1,0), class 1 -> (0,1)
    Y_data[idx * 2] = (label == 0) ? 1.0f : 0.0f;
    Y_data[idx * 2 + 1] = (label == 1) ? 1.0f : 0.0f;
    idx++;
  }
  fclose(file);

  *out_X = sm_create(num_samples, 2);
  *out_Y = sm_create(num_samples, 2);
  if (!*out_X || !*out_Y) {
    log_error("Failed to create matrices");
    sm_destroy(*out_X);
    sm_destroy(*out_Y);
    free(X_data);
    free(Y_data);
    return -1;
  }

  for (size_t i = 0; i < num_samples; i++) {
    sm_set(*out_X, i, 0, X_data[i * 2]);
    sm_set(*out_X, i, 1, X_data[i * 2 + 1]);
    sm_set(*out_Y, i, 0, Y_data[i * 2]);
    sm_set(*out_Y, i, 1, Y_data[i * 2 + 1]);
  }

  free(X_data);
  free(Y_data);
  *out_num_samples = num_samples;
  return 0;
}
