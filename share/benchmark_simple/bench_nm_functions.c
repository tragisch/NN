/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm.h"
#include "simple_bench.h"
#include "sm.h"
#include <stdio.h>
#include <stdlib.h>

void write_bench(char *decription, long long elapsed, long long memory_used);

// Declare benchmark functions
void bench_nm_relu();
void bench_nm_sigmoid();
void bench_nm_tanh();
void bench_nm_softmax();
void bench_nm_loss_cross_entropy();
void bench_nm_loss_mse();
void bench_nm_linear();

int main() {
  bench_nm_relu();

  // Hier können weitere Benchmarks hinzugefügt werden

  return 0;
}

void write_bench(char *decription, long long elapsed, long long memory_used) {
  char formatted_time[64];
  char formatted_memory[64];
  format_duration(elapsed, formatted_time, sizeof(formatted_time));
  format_memory_usage(memory_used, formatted_memory, sizeof(formatted_memory));
  write_double_to_file("share/benchmark_simple/benchmark_results.txt",
                       decription, formatted_time, formatted_memory,
                       nm_backend());
}

void bench_create_random() {
  size_t rows = 20000, cols = 20000;

  long long elapsed, memory_used;
  CPUTIME_RSS(
      {
        FloatMatrix *mat1 = sm_create_random(rows, cols);
        sm_destroy(mat1);
      },
      &elapsed, &memory_used);

  char *description = "sm_create_random (20000, 20000)";
  write_bench(description, elapsed, memory_used);
}

void bench_multiply_by_number() {
  size_t rows = 20000, cols = 20000;
  FloatMatrix *mat1 = sm_create_random(rows, cols);
  float number = 2.0;

  long long elapsed, memory_used;
  CPUTIME_RSS(
      { sm_inplace_multiply_by_number(mat1, number); }, &elapsed, &memory_used);

  char *description = "sm_multiply_by_number (20000, 20000)";
  write_bench(description, elapsed, memory_used);

  sm_destroy(mat1);
}

void bench_nm_relu() {
  size_t rows = 50000, cols = 50000;
  FloatMatrix *mat1 = sm_create_random_he(rows, cols, rows);

  long long elapsed, memory_used;
  CPUTIME_RSS({ nm_relu(mat1); }, &elapsed, &memory_used);

  char *description = "nm_relu (50000, 50000)";
  write_bench(description, elapsed, memory_used);

  sm_destroy(mat1);
}

void bench_nm_sigmoid() {
  size_t rows = 20000, cols = 20000;
  FloatMatrix *mat1 = sm_create_random_he(rows, cols, rows);

  long long elapsed, memory_used;
  CPUTIME_RSS({ nm_sigmoid(mat1); }, &elapsed, &memory_used);

  char *description = "nm_sigmoid (20000, 20000)";
  write_bench(description, elapsed, memory_used);

  sm_destroy(mat1);
}

void bench_nm_tanh() {
  size_t rows = 20000, cols = 20000;
  FloatMatrix *mat1 = sm_create_random_he(rows, cols, rows);

  long long elapsed, memory_used;
  CPUTIME_RSS({ nm_tanh(mat1); }, &elapsed, &memory_used);

  char *description = "nm_tanh (20000, 20000)";
  write_bench(description, elapsed, memory_used);

  sm_destroy(mat1);
}

void bench_nm_softmax() {
  size_t rows = 20000, cols = 20000;
  FloatMatrix *mat1 = sm_create_random_he(rows, cols, rows);

  long long elapsed, memory_used;
  CPUTIME_RSS({ nm_softmax(mat1); }, &elapsed, &memory_used);

  char *description = "nm_softmax (20000, 20000)";
  write_bench(description, elapsed, memory_used);

  sm_destroy(mat1);
}

void bench_nm_loss_cross_entropy() {
  size_t rows = 20000, cols = 20000;
  FloatMatrix *predicted = sm_create_random(rows, cols);
  FloatMatrix *target = sm_create_random(rows, cols);

  long long elapsed, memory_used;
  CPUTIME_RSS(
      { nm_loss_cross_entropy(predicted, target); }, &elapsed, &memory_used);

  char *description = "nm_loss_cross_entropy (20000, 20000)";
  write_bench(description, elapsed, memory_used);

  sm_destroy(predicted);
  sm_destroy(target);
}

void bench_nm_loss_mse() {
  size_t rows = 20000, cols = 20000;
  FloatMatrix *predicted = sm_create_random(rows, cols);
  FloatMatrix *target = sm_create_random(rows, cols);

  long long elapsed, memory_used;
  CPUTIME_RSS({ nm_loss_mse(predicted, target); }, &elapsed, &memory_used);

  char *description = "nm_loss_mse (20000, 20000)";
  write_bench(description, elapsed, memory_used);

  sm_destroy(predicted);
  sm_destroy(target);
}
void bench_nm_linear() {
  size_t rows = 20000, cols = 20000;
  FloatMatrix *input = sm_create_random(rows, cols);
  FloatMatrix *weights = sm_create_random(rows, cols);
  FloatMatrix *bias = sm_create_random(1, cols);
  FloatMatrix *output = NULL;

  long long elapsed, memory_used;
  CPUTIME_RSS(
      { nm_linear(input, weights, bias, &output); }, &elapsed, &memory_used);

  char *description = "nm_linear (20000, 20000)";
  write_bench(description, elapsed, memory_used);

  sm_destroy(output);
  sm_destroy(input);
  sm_destroy(weights);
  sm_destroy(bias);
}
