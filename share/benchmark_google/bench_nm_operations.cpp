extern "C" {
#include "nm.h"
#include "sm.h"
}

#include <benchmark/benchmark.h>

// Benchmark: ReLU activation (typical hidden layer size)
static void BM_nm_relu_128x128(benchmark::State &state) {
  FloatMatrix *mat = sm_create(128, 128);
  if (!mat || !mat->values) {
    state.SkipWithError("Failed to allocate matrix");
    return;
  }

  for (size_t i = 0; i < mat->rows * mat->cols; ++i)
    mat->values[i] = (float)(i % 3) - 1.0f;

  for (auto _ : state) {
    nm_relu(mat);
    benchmark::DoNotOptimize(mat->values);
  }

  state.SetItemsProcessed(state.iterations() * (int64_t)mat->rows * mat->cols);
  sm_destroy(mat);
}

// Benchmark: Softmax activation (output layer)
static void BM_nm_softmax_128x10(benchmark::State &state) {
  FloatMatrix *mat = sm_create(128, 10);
  if (!mat || !mat->values) {
    state.SkipWithError("Failed to allocate matrix");
    return;
  }

  for (size_t i = 0; i < mat->rows * mat->cols; ++i)
    mat->values[i] = (float)i * 0.01f;

  for (auto _ : state) {
    nm_softmax(mat);
    benchmark::DoNotOptimize(mat->values);
  }

  state.SetItemsProcessed(state.iterations() * (int64_t)mat->rows * mat->cols);
  sm_destroy(mat);
}

// Benchmark: Sum rows (bias gradient computation)
static void BM_nm_sum_rows_128x10(benchmark::State &state) {
  FloatMatrix *mat = sm_create(128, 10);
  if (!mat || !mat->values) {
    state.SkipWithError("Failed to allocate matrix");
    return;
  }

  for (size_t i = 0; i < mat->rows * mat->cols; ++i)
    mat->values[i] = 1.0f;

  for (auto _ : state) {
    FloatMatrix *result = nullptr;
    if (nm_sum_rows(mat, &result) != NM_STATUS_OK || !result) {
      state.SkipWithError("nm_sum_rows failed");
      break;
    }
    benchmark::DoNotOptimize(result);
    sm_destroy(result);
  }

  state.SetItemsProcessed(state.iterations() * (int64_t)mat->rows * mat->cols);
  sm_destroy(mat);
}

// Benchmark: Add rowwise (bias addition in forward pass)
static void BM_nm_add_rowwise_128x128(benchmark::State &state) {
  FloatMatrix *mat = sm_create(128, 128);
  FloatMatrix *row = sm_create(1, 128);

  if (!mat || !row || !mat->values || !row->values) {
    state.SkipWithError("Failed to allocate matrices");
    sm_destroy(mat);
    sm_destroy(row);
    return;
  }

  for (size_t i = 0; i < mat->rows * mat->cols; ++i)
    mat->values[i] = 1.0f;
  for (size_t i = 0; i < row->cols; ++i)
    row->values[i] = 0.1f;

  for (auto _ : state) {
    nm_add_rowwise(mat, row);
    benchmark::DoNotOptimize(mat->values);
  }

  state.SetItemsProcessed(state.iterations() * (int64_t)mat->rows * mat->cols);
  sm_destroy(mat);
  sm_destroy(row);
}

// Benchmark: Linear layer forward (typical hidden layer)
static void BM_nm_linear_128x784_to_128(benchmark::State &state) {
  FloatMatrix *input = sm_create(128, 784);
  FloatMatrix *weights = sm_create(784, 128);
  FloatMatrix *bias = sm_create(1, 128);

  if (!input || !weights || !bias) {
    state.SkipWithError("Failed to allocate matrices");
    sm_destroy(input);
    sm_destroy(weights);
    sm_destroy(bias);
    return;
  }

  for (size_t i = 0; i < input->rows * input->cols; ++i)
    input->values[i] = 1.0f;
  for (size_t i = 0; i < weights->rows * weights->cols; ++i)
    weights->values[i] = 0.01f;
  for (size_t i = 0; i < bias->cols; ++i)
    bias->values[i] = 0.1f;

  for (auto _ : state) {
    FloatMatrix *result = nullptr;
    if (nm_linear(input, weights, bias, &result) != NM_STATUS_OK || !result) {
      state.SkipWithError("nm_linear failed");
      break;
    }
    benchmark::DoNotOptimize(result->values);
    sm_destroy(result);
  }

  state.SetItemsProcessed(state.iterations() * (int64_t)input->rows);
  sm_destroy(input);
  sm_destroy(weights);
  sm_destroy(bias);
}

BENCHMARK(BM_nm_relu_128x128);
BENCHMARK(BM_nm_softmax_128x10);
BENCHMARK(BM_nm_sum_rows_128x10);
BENCHMARK(BM_nm_add_rowwise_128x128);
BENCHMARK(BM_nm_linear_128x784_to_128);

BENCHMARK_MAIN();
