/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

// Minimal MNIST example using nm_frontend and nm_engine_c

#include "mnist_data.h"
#include "nm_engine_c.h"
#include "nm_frontend.h"
#include <log.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct {
  FloatMatrix *X_train;
  FloatMatrix *Y_train;
  FloatMatrix *X_test;
  FloatMatrix *Y_test;
} MnistDataset;

static void mnist_dataset_destroy(MnistDataset *ds) {
  if (!ds)
    return;
  sm_destroy(ds->X_train);
  sm_destroy(ds->Y_train);
  sm_destroy(ds->X_test);
  sm_destroy(ds->Y_test);
  *ds = (MnistDataset){0};
}

static bool mnist_dataset_load(MnistDataset *ds, size_t train_size,
                               size_t test_size, size_t num_classes) {
  if (!ds)
    return false;

  *ds = (MnistDataset){0};
  ds->X_train =
      load_mnist_images("examples/mnist_2/train-images-idx3-ubyte", train_size);
  ds->Y_train = load_mnist_labels_onehot(
      "examples/mnist_2/train-labels-idx1-ubyte", train_size, num_classes);
  ds->X_test =
      load_mnist_images("examples/mnist_2/t10k-images-idx3-ubyte", test_size);
  ds->Y_test = load_mnist_labels_onehot("examples/mnist_2/t10k-labels-idx1-ubyte",
                                        test_size, num_classes);

  if (!ds->X_train || !ds->Y_train || !ds->X_test || !ds->Y_test) {
    if (!ds->X_train)
      ds->X_train = load_mnist_images("examples/mnist/train-images-idx3-ubyte",
                                      train_size);
    if (!ds->Y_train)
      ds->Y_train = load_mnist_labels_onehot(
          "examples/mnist/train-labels-idx1-ubyte", train_size, num_classes);
    if (!ds->X_test)
      ds->X_test = load_mnist_images("examples/mnist/t10k-images-idx3-ubyte",
                                     test_size);
    if (!ds->Y_test)
      ds->Y_test = load_mnist_labels_onehot(
          "examples/mnist/t10k-labels-idx1-ubyte", test_size, num_classes);
  }

  if (!ds->X_train || !ds->Y_train || !ds->X_test || !ds->Y_test) {
    mnist_dataset_destroy(ds);
    return false;
  }
  return true;
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

int main(void) {
  log_set_level(LOG_INFO);

  const size_t num_classes = 10;
  const size_t input_dim = 28 * 28;
  const size_t hidden_dim = 128;
  const size_t train_size = 60000;
  const size_t test_size = 10000;
  const size_t batch_size = 128;
  const float learning_rate = 0.1f;
  const size_t epochs = 3;

  NmStatus s;
  NmFrontendModel *model = NULL;
  NmEngineC *engine = NULL;
  FloatMatrix *y_pred = NULL;
  MnistDataset ds = {0};

  // 1) Load data
  if (!mnist_dataset_load(&ds, train_size, test_size, num_classes)) {
    log_error("Failed to load MNIST data");
    return EXIT_FAILURE;
  }

  // 2) Define model: 784 -> 128 (ReLU) -> 10 (Softmax)
  model = nm_frontend_create((NmShape){.ndim = 1, .dims = {input_dim}});
  if (!model) {
    log_error("Failed to create model");
    mnist_dataset_destroy(&ds);
    return EXIT_FAILURE;
  }

  s = nm_frontend_add_dense(model, &(NmDenseConfig){
                                       .units = hidden_dim,
                                       .activation = NM_ACT_RELU,
                                   });
  if (s == NM_STATUS_OK) {
    s = nm_frontend_add_dense(model, &(NmDenseConfig){
                                         .units = num_classes,
                                         .activation = NM_ACT_SOFTMAX,
                                     });
  }
  if (s == NM_STATUS_OK) {
    s = nm_frontend_finalize(model);
  }
  if (s != NM_STATUS_OK) {
    log_error("Failed to build model (status=%d)", (int)s);
    nm_frontend_free(model);
    mnist_dataset_destroy(&ds);
    return EXIT_FAILURE;
  }

  // 3) Train via C engine
  engine = nm_engine_c_create(NULL);
  if (!engine) {
    log_error("Failed to create C engine");
    nm_frontend_free(model);
    mnist_dataset_destroy(&ds);
    return EXIT_FAILURE;
  }

  s = nm_engine_c_prepare(engine, model);
  if (s != NM_STATUS_OK) {
    log_error("Engine prepare failed (status=%d)", (int)s);
    nm_engine_c_free(engine);
    nm_frontend_free(model);
    mnist_dataset_destroy(&ds);
    return EXIT_FAILURE;
  }

  NmTrainConfig fit_cfg = {
      .batch_size = batch_size,
      .learning_rate = learning_rate,
      .epochs = epochs,
      .verbose = false,
  };

  s = nm_engine_c_fit(engine, model, ds.X_train, ds.Y_train, &fit_cfg);
  if (s != NM_STATUS_OK) {
    log_error("Training failed (status=%d)", (int)s);
    nm_engine_c_free(engine);
    nm_frontend_free(model);
    mnist_dataset_destroy(&ds);
    return EXIT_FAILURE;
  }

  // 4) Evaluate
  s = nm_engine_c_predict(engine, model, ds.X_test, &y_pred);
  if (s == NM_STATUS_OK && y_pred) {
    float test_acc = compute_accuracy(y_pred, ds.Y_test);
    log_info("Final Test Accuracy: %.2f%%", test_acc * 100.0f);
  } else {
    log_error("Evaluation failed (status=%d)", (int)s);
  }

  sm_destroy(y_pred);
  nm_engine_c_free(engine);
  nm_frontend_free(model);
  mnist_dataset_destroy(&ds);

  return s == NM_STATUS_OK ? EXIT_SUCCESS : EXIT_FAILURE;
}

