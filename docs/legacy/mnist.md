# Example: MNIST

!!! warning "Legacy API"
    This page describes the legacy `nm.h` API, which stays available but is not
    the recommended entry point. New programs should use `nm_api.h`; see
    [API Overview](../api/overview.md).

Reference implementation: `examples/mnist/main.c`

This snippet is intentionally complete: load data, configure the model, train,
evaluate, and clean up. It demonstrates that the API is easy to use by default,
while still allowing expert-level tuning.

## Full end-to-end snippet

```c
#include "nm.h"
#include "mnist_data.h"
#include <log.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  log_set_level(LOG_INFO);
  nm_set_seed(42);

  // Dataset / model dimensions
  const size_t input_dim = 28 * 28;
  const size_t num_classes = 10;
  const size_t train_size = 60000;
  const size_t test_size = 10000;

  // Training hyperparameters
  const size_t hidden_dim = 128;
  const size_t batch_size = 128;
  const float learning_rate = 0.02f;
  const size_t epochs = 10;

  // 1) Load MNIST data
  FloatMatrix *X_train =
      load_mnist_images("examples/mnist/train-images-idx3-ubyte", train_size);
  FloatMatrix *Y_train =
      load_mnist_labels_onehot("examples/mnist/train-labels-idx1-ubyte",
                               train_size, num_classes);
  FloatMatrix *X_test =
      load_mnist_images("examples/mnist/t10k-images-idx3-ubyte", test_size);
  FloatMatrix *Y_test =
      load_mnist_labels_onehot("examples/mnist/t10k-labels-idx1-ubyte",
                               test_size, num_classes);

  if (!X_train || !Y_train || !X_test || !Y_test) {
    log_error("Failed to load MNIST data");
    sm_destroy(X_train);
    sm_destroy(Y_train);
    sm_destroy(X_test);
    sm_destroy(Y_test);
    return EXIT_FAILURE;
  }

  // 2) Build a classifier
  NmShape input_shape = {.ndim = 1, .dims = {input_dim, 1, 1}};
  NmModel *model = nm_create(input_shape);
  if (!model) {
    log_error("Failed to create model");
    sm_destroy(X_train);
    sm_destroy(Y_train);
    sm_destroy(X_test);
    sm_destroy(Y_test);
    return EXIT_FAILURE;
  }

  NmDenseConfig hidden = {
      .units = hidden_dim,
      .activation = NM_ACT_RELU,
  };
  NmDenseConfig output = {
      .units = num_classes,
      .activation = NM_ACT_SOFTMAX,
  };

  NmStatus s = nm_add_dense(model, &hidden);
  if (s == NM_STATUS_OK) {
    s = nm_add_dense(model, &output);
  }

  // 3) Compile
  NmCompileConfig expert_cfg = nm_compile_defaults();
  expert_cfg.loss_type = NM_LOSS_CROSS_ENTROPY;
  if (s == NM_STATUS_OK) {
    s = nm_compile(model, &expert_cfg);
  }
  if (s != NM_STATUS_OK) {
    log_error("Invalid compile config (status=%d)", (int)s);
    nm_free(model);
    sm_destroy(X_train);
    sm_destroy(Y_train);
    sm_destroy(X_test);
    sm_destroy(Y_test);
    return EXIT_FAILURE;
  }

  // 4) Train
  NmTrainConfig train_cfg = nm_train_defaults();
  train_cfg.batch_size = batch_size;
  train_cfg.learning_rate = learning_rate;
  train_cfg.epochs = epochs;
  train_cfg.verbose = true;

  s = nm_fit(model, X_train, Y_train, &train_cfg);
  if (s != NM_STATUS_OK) {
    log_error("Training failed (status=%d)", (int)s);
    nm_free(model);
    sm_destroy(X_train);
    sm_destroy(Y_train);
    sm_destroy(X_test);
    sm_destroy(Y_test);
    return EXIT_FAILURE;
  }

  // 5) Evaluate on test set
  float test_acc = 0.0f;
  FloatMatrix *y_pred = NULL;
  s = nm_predict(model, X_test, &y_pred);
  if (s == NM_STATUS_OK) {
    s = nm_accuracy(y_pred, Y_test, &test_acc);
  }
  sm_destroy(y_pred);

  if (s != NM_STATUS_OK) {
    log_error("Evaluation failed (status=%d)", (int)s);
    nm_free(model);
    sm_destroy(X_train);
    sm_destroy(Y_train);
    sm_destroy(X_test);
    sm_destroy(Y_test);
    return EXIT_FAILURE;
  }

  log_info("Final Test Accuracy: %.2f%%", test_acc * 100.0f);

  // 6) Cleanup
  sm_destroy(X_train);
  sm_destroy(Y_train);
  sm_destroy(X_test);
  sm_destroy(Y_test);
  nm_free(model);
  return EXIT_SUCCESS;
}
```

## Why this is a good API showcase

- **Builder path:** clear model/training setup via one model handle plus config structs.
- **Expert path:** override compile and training settings field-by-field.
- **Production-safe:** clear error checks and cleanup on every failure path.
