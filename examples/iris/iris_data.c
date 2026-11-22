/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "iris_data.h"
#include <log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

static FILE *open_iris_file(const char *filename) {
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

int load_iris_dataset(const char *filepath, FloatMatrix **out_X,
                      FloatMatrix **out_Y, size_t *out_num_samples) {
  if (!filepath || !out_X || !out_Y || !out_num_samples)
    return -1;

  FILE *file = open_iris_file(filepath);
  if (!file) {
    log_error("Failed to open Iris data file: %s", filepath);
    return -1;
  }

  // First pass: count valid lines
  size_t num_samples = 0;
  char line[256];
  while (fgets(line, sizeof(line), file)) {
    // Skip empty lines
    if (strlen(line) <= 1)
      continue;
    num_samples++;
  }

  if (num_samples == 0) {
    log_error("No data found in file");
    fclose(file);
    return -1;
  }

  // Allocate arrays for features and labels
  float *X_data = (float *)malloc(num_samples * 4 * sizeof(float));
  float *Y_data = (float *)malloc(num_samples * 3 * sizeof(float));
  if (!X_data || !Y_data) {
    log_error("Memory allocation failed");
    free(X_data);
    free(Y_data);
    fclose(file);
    return -1;
  }

  // Second pass: read data
  rewind(file);
  size_t idx = 0;
  while (fgets(line, sizeof(line), file) && idx < num_samples) {
    // Skip empty lines
    if (strlen(line) <= 1)
      continue;

    float f1, f2, f3, f4;
    char species[50];

    // Parse CSV: sepal_length,sepal_width,petal_length,petal_width,species
    if (sscanf(line, "%f,%f,%f,%f,%49[^\n]", &f1, &f2, &f3, &f4, species) != 5) {
      log_warn("Skipping malformed line: %s", line);
      continue;
    }

    // Store features
    X_data[idx * 4 + 0] = f1; // sepal length
    X_data[idx * 4 + 1] = f2; // sepal width
    X_data[idx * 4 + 2] = f3; // petal length
    X_data[idx * 4 + 3] = f4; // petal width

    // One-hot encode species
    Y_data[idx * 3 + 0] = 0.0f;
    Y_data[idx * 3 + 1] = 0.0f;
    Y_data[idx * 3 + 2] = 0.0f;

    if (strstr(species, "Iris-setosa")) {
      Y_data[idx * 3 + 0] = 1.0f;
    } else if (strstr(species, "Iris-versicolor")) {
      Y_data[idx * 3 + 1] = 1.0f;
    } else if (strstr(species, "Iris-virginica")) {
      Y_data[idx * 3 + 2] = 1.0f;
    } else {
      log_warn("Unknown species: %s", species);
    }

    idx++;
  }

  fclose(file);

  // Create matrices
  *out_X = sm_create_with_values(idx, 4, X_data);
  *out_Y = sm_create_with_values(idx, 3, Y_data);

  free(X_data);
  free(Y_data);

  if (!*out_X || !*out_Y) {
    log_error("Failed to create matrices");
    sm_destroy(*out_X);
    sm_destroy(*out_Y);
    return -1;
  }

  *out_num_samples = idx;
  log_info("Loaded %zu Iris samples (4 features, 3 classes)", idx);

  return 0;
}
