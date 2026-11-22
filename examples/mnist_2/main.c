/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

// MNIST classification using nm_frontend, nm_engine_c, and GraphIR export

#include "argtable3.h"
#include "mnist_data.h"
#include "nm_engine_c.h"
#include "nm_export_graphir.h"
#include "nm_frontend.h"
#include <log.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double get_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static float compute_accuracy(const FloatMatrix *y_pred,
                              const FloatMatrix *y_true) {
  if (!y_pred || !y_true || y_pred->rows != y_true->rows ||
      y_pred->cols != y_true->cols)
    return 0.0f;

  size_t correct = 0;
  for (size_t i = 0; i < y_pred->rows; ++i) {
    size_t pred_class = 0;
    float pred_max = y_pred->values[i * y_pred->cols];
    size_t true_class = 0;
    float true_max = y_true->values[i * y_true->cols];
    for (size_t c = 1; c < y_pred->cols; ++c) {
      float pv = y_pred->values[i * y_pred->cols + c];
      if (pv > pred_max) {
        pred_max = pv;
        pred_class = c;
      }
      float tv = y_true->values[i * y_true->cols + c];
      if (tv > true_max) {
        true_max = tv;
        true_class = c;
      }
    }
    if (pred_class == true_class)
      correct++;
  }
  return (float)correct / (float)y_pred->rows;
}

static bool export_binary_file(const char *path, const void *data,
                               size_t size_bytes) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    log_error("Failed to open %s for writing", path);
    return false;
  }
  size_t written = fwrite(data, 1, size_bytes, f);
  fclose(f);
  return written == size_bytes;
}

int main(int argc, char *argv[]) {
  /* ---- CLI Options ------------------------------------------------ */
  struct arg_int *opt_epochs =
      arg_int0(NULL, "epochs", "<n>", "Training epochs       (default: 5)");
  struct arg_dbl *opt_lr =
      arg_dbl0(NULL, "lr", "<f>", "Learning rate         (default: 0.1)");
  struct arg_int *opt_batch =
      arg_int0(NULL, "batch", "<n>", "Batch size            (default: 128)");
  struct arg_int *opt_hidden =
      arg_int0(NULL, "hidden", "<n>", "Hidden units          (default: 128)");
  struct arg_str *opt_export = arg_str0(
      NULL, "export", "<path>", "Export GraphIR JSON   (default: mnist.graphir.json)");
  struct arg_int *opt_eval_samples = arg_int0(
      NULL, "eval-samples", "<n>", "Evaluation samples    (default: 10000)");
  struct arg_int *opt_bench_iters = arg_int0(
      NULL, "bench-iters", "<n>", "Benchmark iterations  (default: 20)");
  struct arg_lit *opt_verbose =
      arg_lit0("v", "verbose", "Verbose training output");
  struct arg_lit *opt_help = arg_lit0("h", "help", "Show this help");
  struct arg_end *end = arg_end(20);

  void *argtable[] = {opt_epochs,       opt_lr,          opt_batch,
                      opt_hidden,       opt_export,      opt_eval_samples,
                      opt_bench_iters,  opt_verbose,     opt_help,
                      end};

  int nerrors = arg_parse(argc, argv, argtable);
  if (opt_help->count > 0) {
    printf("Usage: mnist");
    arg_print_syntax(stdout, argtable, "\n");
    arg_print_glossary(stdout, argtable, "  %-28s %s\n");
    arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));
    return EXIT_SUCCESS;
  }
  if (nerrors > 0) {
    arg_print_errors(stderr, end, "mnist");
    arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));
    return EXIT_FAILURE;
  }

  const size_t hidden_dim =
      opt_hidden->count ? (size_t)opt_hidden->ival[0] : 128;
  const size_t batch_size =
      opt_batch->count ? (size_t)opt_batch->ival[0] : 128;
  const float learning_rate =
      opt_lr->count ? (float)opt_lr->dval[0] : 0.1f;
  const size_t epochs = opt_epochs->count ? (size_t)opt_epochs->ival[0] : 5;
  const char *export_path =
      opt_export->count ? opt_export->sval[0] : "mnist.graphir.json";
  const size_t eval_samples =
      opt_eval_samples->count ? (size_t)opt_eval_samples->ival[0] : 10000;
  const size_t bench_iters =
      opt_bench_iters->count ? (size_t)opt_bench_iters->ival[0] : 20;
  const bool verbose = opt_verbose->count > 0;
  arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));

  log_set_level(LOG_INFO);

  const size_t num_classes = 10;
  const size_t input_dim = 28 * 28;
  const size_t train_size = 60000;
  const size_t test_size = 10000;

  int exit_code = EXIT_FAILURE;
  FloatMatrix *X_train = NULL;
  FloatMatrix *Y_train = NULL;
  FloatMatrix *X_test = NULL;
  FloatMatrix *Y_test = NULL;
  NmFrontendModel *model = NULL;
  NmEngineC *engine = NULL;
  FloatMatrix *prediction = NULL;
  FloatMatrix *X_eval = NULL;
  FloatMatrix *Y_eval = NULL;

  /* ---- Load Dataset ------------------------------------------------ */
  log_info("Loading MNIST dataset...");
  X_train =
      load_mnist_images("examples/mnist_2/train-images-idx3-ubyte", train_size);
  Y_train = load_mnist_labels_onehot(
      "examples/mnist_2/train-labels-idx1-ubyte", train_size, num_classes);
  X_test =
      load_mnist_images("examples/mnist_2/t10k-images-idx3-ubyte", test_size);
  Y_test = load_mnist_labels_onehot(
      "examples/mnist_2/t10k-labels-idx1-ubyte", test_size, num_classes);

  if (!X_train || !Y_train || !X_test || !Y_test) {
    if (!X_train)
      X_train = load_mnist_images("examples/mnist/train-images-idx3-ubyte",
                                  train_size);
    if (!Y_train)
      Y_train = load_mnist_labels_onehot("examples/mnist/train-labels-idx1-ubyte",
                                         train_size, num_classes);
    if (!X_test)
      X_test = load_mnist_images("examples/mnist/t10k-images-idx3-ubyte",
                                 test_size);
    if (!Y_test)
      Y_test = load_mnist_labels_onehot("examples/mnist/t10k-labels-idx1-ubyte",
                                        test_size, num_classes);
  }

  if (!X_train || !Y_train || !X_test || !Y_test) {
    log_error("Failed to load MNIST data files");
    goto cleanup;
  }

  /* Slice evaluation set to requested eval_samples */
  size_t actual_eval = (eval_samples < test_size) ? eval_samples : test_size;
  X_eval = sm_create_with_values(actual_eval, input_dim, X_test->values);
  Y_eval = sm_create_with_values(actual_eval, num_classes, Y_test->values);
  if (!X_eval || !Y_eval) {
    log_error("Failed to allocate evaluation dataset");
    goto cleanup;
  }

  /* ---- Build Model with nm_frontend -------------------------------- */
  log_info("Constructing Frontend Model (784 -> %zu -> 10)...", hidden_dim);
  model = nm_frontend_create((NmShape){.ndim = 1, .dims = {input_dim}});
  if (!model) {
    log_error("Failed to create frontend model");
    goto cleanup;
  }

  NmStatus status = nm_frontend_add_dense(
      model, &(NmDenseConfig){.units = hidden_dim, .activation = NM_ACT_RELU});
  if (status == NM_STATUS_OK) {
    status = nm_frontend_add_dense(
        model,
        &(NmDenseConfig){.units = num_classes, .activation = NM_ACT_SOFTMAX});
  }
  if (status == NM_STATUS_OK) {
    status = nm_frontend_finalize(model);
  }
  if (status != NM_STATUS_OK) {
    log_error("Frontend model construction failed (status=%d)", (int)status);
    goto cleanup;
  }

  /* ---- Train with nm_engine_c -------------------------------------- */
  engine = nm_engine_c_create(NULL);
  if (!engine) {
    log_error("Failed to create C engine");
    goto cleanup;
  }

  status = nm_engine_c_prepare(engine, model);
  if (status != NM_STATUS_OK) {
    log_error("Engine prepare failed (status=%d)", (int)status);
    goto cleanup;
  }

  NmTrainConfig train_cfg = {
      .batch_size = batch_size,
      .learning_rate = learning_rate,
      .epochs = epochs,
      .verbose = verbose,
  };

  log_info("Training model with C engine (epochs=%zu, batch=%zu, lr=%.3f)...",
           epochs, batch_size, learning_rate);
  double train_start = get_time_ms();
  status = nm_engine_c_fit(engine, model, X_train, Y_train, &train_cfg);
  double train_time_ms = get_time_ms() - train_start;

  if (status != NM_STATUS_OK) {
    log_error("Training failed (status=%d)", (int)status);
    goto cleanup;
  }
  log_info("Training completed in %.2f s", train_time_ms / 1000.0);

  /* ---- Inference & Accuracy ---------------------------------------- */
  status = nm_engine_c_predict(engine, model, X_eval, &prediction);
  if (status != NM_STATUS_OK || !prediction) {
    log_error("Inference failed (status=%d)", (int)status);
    goto cleanup;
  }

  float accuracy = compute_accuracy(prediction, Y_eval);
  log_info("nm_lib (C Engine) Test Accuracy (%zu samples): %.2f%%",
           actual_eval, accuracy * 100.0f);

  /* ---- Benchmark Inference (nm_lib / C Engine) --------------------- */
  log_info("Benchmarking nm_lib C Engine inference (%zu iterations)...",
           bench_iters);
  double total_eval_ms = 0.0;
  for (size_t iter = 0; iter < bench_iters; ++iter) {
    FloatMatrix *tmp_pred = NULL;
    double iter_start = get_time_ms();
    nm_engine_c_predict(engine, model, X_eval, &tmp_pred);
    double iter_time = get_time_ms() - iter_start;
    total_eval_ms += iter_time;
    sm_destroy(tmp_pred);
  }
  double mean_latency_ms = total_eval_ms / (double)bench_iters;
  double throughput = ((double)actual_eval * (double)bench_iters) /
                      (total_eval_ms / 1000.0);

  printf("\n=== nm_lib (C Engine) Benchmark Results ===\n");
  printf("  Samples per batch : %zu\n", actual_eval);
  printf("  Mean Batch Latency: %.2f ms\n", mean_latency_ms);
  printf("  Latency per sample: %.4f ms\n", mean_latency_ms / (double)actual_eval);
  printf("  Throughput        : %.1f samples/sec\n", throughput);
  printf("  Accuracy          : %.2f%%\n\n", accuracy * 100.0f);

  /* ---- Export GraphIR JSON ----------------------------------------- */
  log_info("Exporting GraphIR JSON to %s (batch_size=%zu)...", export_path,
           actual_eval);
  NmGraphIrExportOptions export_opts = {.batch_size = actual_eval};
  status = nm_export_graphir_json(model, &export_opts, export_path);
  if (status != NM_STATUS_OK) {
    log_error("GraphIR export failed (status=%d)", (int)status);
    goto cleanup;
  }
  log_info("GraphIR export successful: %s", export_path);

  /* ---- Export Evaluation Data & Predictions for Rust / Core AI ----- */
  export_binary_file("mnist_test_inputs.bin", X_eval->values,
                     actual_eval * input_dim * sizeof(float));
  export_binary_file("mnist_c_predictions.bin", prediction->values,
                     actual_eval * num_classes * sizeof(float));

  uint8_t *labels_u8 = (uint8_t *)malloc(actual_eval);
  if (labels_u8) {
    for (size_t i = 0; i < actual_eval; ++i) {
      uint8_t label = 0;
      for (size_t c = 0; c < num_classes; ++c) {
        if (Y_eval->values[i * num_classes + c] > 0.5f) {
          label = (uint8_t)c;
          break;
        }
      }
      labels_u8[i] = label;
    }
    export_binary_file("mnist_test_labels.bin", labels_u8, actual_eval);
    free(labels_u8);
  }

  exit_code = EXIT_SUCCESS;

cleanup:
  sm_destroy(prediction);
  sm_destroy(X_eval);
  sm_destroy(Y_eval);
  nm_engine_c_free(engine);
  nm_frontend_free(model);
  sm_destroy(X_train);
  sm_destroy(Y_train);
  sm_destroy(X_test);
  sm_destroy(Y_test);
  return exit_code;
}

