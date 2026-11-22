/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

// A simple example demonstrating how to train a feedforward neural network on
// the MNIST dataset

#include "argtable3.h"
#include "mnist_data.h"
#include "nm_api.h"
#include <log.h>
#include <stdio.h>
#include <stdlib.h>

static int on_epoch(const NmProgress *progress, void *user) {
  const bool verbose = *(const bool *)user;
  if (verbose)
    log_info("Epoch %zu/%zu - loss %.4f", progress->epoch, progress->epochs,
             progress->loss);
  return 0;
}

int main(int argc, char *argv[]) {
  /* ---- CLI -------------------------------------------------------- */
  struct arg_int *opt_epochs =
      arg_int0(NULL, "epochs", "<n>", "Training epochs       (default: 10)");
  struct arg_dbl *opt_lr =
      arg_dbl0(NULL, "lr", "<f>", "Learning rate         (default: 0.02)");
  struct arg_int *opt_batch =
      arg_int0(NULL, "batch", "<n>", "Batch size            (default: 128)");
  struct arg_int *opt_hidden =
      arg_int0(NULL, "hidden", "<n>", "Hidden units          (default: 128)");
  struct arg_str *opt_save =
      arg_str0(NULL, "save", "<dir>", "Save model directory after training");
  struct arg_str *opt_load =
      arg_str0(NULL, "load", "<dir>", "Load model directory and skip training");
  struct arg_lit *opt_verbose =
      arg_lit0("v", "verbose", "Verbose training output");
  struct arg_lit *opt_help = arg_lit0("h", "help", "Show this help");
  struct arg_end *end = arg_end(20);

  void *argtable[] = {opt_epochs, opt_lr,      opt_batch, opt_hidden, opt_save,
                      opt_load,   opt_verbose, opt_help,  end};

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
  const size_t batch_size = opt_batch->count ? (size_t)opt_batch->ival[0] : 128;
  const float learning_rate = opt_lr->count ? (float)opt_lr->dval[0] : 0.02f;
  const size_t epochs = opt_epochs->count ? (size_t)opt_epochs->ival[0] : 10;
  const bool verbose = opt_verbose->count > 0;
  const char *save_path = opt_save->count ? opt_save->sval[0] : NULL;
  const char *load_path = opt_load->count ? opt_load->sval[0] : NULL;
  arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));

  /* ---- Setup ------------------------------------------------------ */
  log_set_level(LOG_INFO);

  const size_t num_classes = 10;
  const size_t input_dim = 28 * 28;
  const size_t train_size = 60000;
  const size_t test_size = 10000;

  int exit_code = EXIT_FAILURE;
  NmStatus s = NM_STATUS_OK;
  NmNetwork *net = NULL;
  NmSpec *spec = NULL;

  FloatMatrix *X_train =
      load_mnist_images("examples/mnist/train-images-idx3-ubyte", train_size);
  FloatMatrix *Y_train = load_mnist_labels_onehot(
      "examples/mnist/train-labels-idx1-ubyte", train_size, num_classes);
  FloatMatrix *X_test =
      load_mnist_images("examples/mnist/t10k-images-idx3-ubyte", test_size);
  FloatMatrix *Y_test = load_mnist_labels_onehot(
      "examples/mnist/t10k-labels-idx1-ubyte", test_size, num_classes);

  if (!X_train || !Y_train || !X_test || !Y_test) {
    log_error("Failed to load MNIST data");
    goto cleanup;
  }

  /* ---- Model construction ----------------------------------------- */
  if (load_path) {
    log_info("Loading model from %s", load_path);
    net = nm_network_load(load_path);
    if (!net) {
      log_error("Failed to load model: %s", nm_last_error());
      goto cleanup;
    }
  } else {
    spec = nm_spec_new((NmShape){.ndim = 1, .dims = {input_dim}});
    if (!spec) {
      log_error("Failed to create model spec: %s", nm_last_error());
      goto cleanup;
    }

    s = nm_spec_dense(spec, hidden_dim, NM_ACT_RELU);
    if (s == NM_STATUS_OK) {
      s = nm_spec_dense(spec, num_classes, NM_ACT_SOFTMAX);
    }
    if (s != NM_STATUS_OK) {
      log_error("Failed to describe model: %s", nm_last_error());
      goto cleanup;
    }

    NmBuildOptions build = nm_build_options_default();
    build.seed = 42;
    net = nm_network_build(spec, &build);
    nm_spec_free(spec);
    spec = NULL;
    if (!net) {
      log_error("Failed to build model: %s", nm_last_error());
      goto cleanup;
    }

    NmFitOptions fit = nm_fit_options_default();
    fit.loss = NM_LOSS_CROSS_ENTROPY;
    fit.batch_size = batch_size;
    fit.learning_rate = learning_rate;
    fit.epochs = epochs;
    fit.shuffle = true;

    NmTensor x_train =
        nm_tensor_2d(X_train->values, X_train->rows, X_train->cols);
    NmTensor y_train =
        nm_tensor_2d(Y_train->values, Y_train->rows, Y_train->cols);
    s = nm_network_fit(net, x_train, &y_train, &fit, verbose ? on_epoch : NULL,
                       (void *)&verbose);
    if (s != NM_STATUS_OK) {
      log_error("Training failed: %s", nm_last_error());
      goto cleanup;
    }

    if (save_path) {
      log_info("Saving model to %s", save_path);
      NmStatus save_s = nm_network_save(net, save_path);
      if (save_s != NM_STATUS_OK)
        log_warn("Save failed: %s \xe2\x80\x94 continuing", nm_last_error());
    }
  }

  /* ---- Evaluation ------------------------------------------------- */
  NmTensor x_test = nm_tensor_2d(X_test->values, X_test->rows, X_test->cols);
  NmTensor y_test = nm_tensor_2d(Y_test->values, Y_test->rows, Y_test->cols);
  NmMetrics metrics;
  s = nm_network_evaluate(net, x_test, y_test, NM_LOSS_CROSS_ENTROPY, &metrics);
  if (s != NM_STATUS_OK) {
    log_error("Evaluation failed: %s", nm_last_error());
    goto cleanup;
  }

  log_info("Final Test Accuracy: %.2f%%", metrics.accuracy * 100.0f);
  exit_code = EXIT_SUCCESS;

cleanup:
  sm_destroy(X_train);
  sm_destroy(Y_train);
  sm_destroy(X_test);
  sm_destroy(Y_test);
  nm_spec_free(spec);
  nm_network_free(net);

  return exit_code;
}
