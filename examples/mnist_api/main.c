/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* MNIST with the user API: about fifteen lines of user code. Run from the
 * repository root: bazel run //examples/mnist_api:mnist_api */
#include "mnist_data.h"
#include "nm_api.h"
#include <stdio.h>

static NmTensor view(const FloatMatrix *m) {
  return nm_tensor_2d(m->values, m->rows, m->cols);
}

static int on_epoch(const NmProgress *p, void *user) {
  (void)user;
  printf("epoch %2zu/%zu  loss %.4f  val %.4f\n", p->epoch, p->epochs, p->loss, p->val_loss);
  return 0;
}

int main(void) {
  FloatMatrix *x = load_mnist_images("examples/mnist/train-images-idx3-ubyte", 10000);
  FloatMatrix *y = load_mnist_labels_onehot("examples/mnist/train-labels-idx1-ubyte", 10000, 10);
  FloatMatrix *xt = load_mnist_images("examples/mnist/t10k-images-idx3-ubyte", 2000);
  FloatMatrix *yt = load_mnist_labels_onehot("examples/mnist/t10k-labels-idx1-ubyte", 2000, 10);
  if (!x || !y || !xt || !yt) return 1;
  /* --- the user code starts here --- */
  NmSpec *spec = nm_spec_new((NmShape){.ndim = 1, .dims = {784}});
  nm_spec_dense(spec, 128, NM_ACT_RELU);
  nm_spec_dense(spec, 10, NM_ACT_SOFTMAX);
  NmNetwork *net = nm_network_build(spec, NULL);
  NmFitOptions fit = nm_fit_options_default();
  fit.epochs = 10; fit.validation_split = 0.1f;
  NmTensor targets = view(y);
  nm_network_fit(net, view(x), &targets, &fit, on_epoch, NULL);
  NmMetrics m;
  nm_network_evaluate(net, view(xt), view(yt), NM_LOSS_CROSS_ENTROPY, &m);
  printf("test accuracy %.2f%%\n", 100.0 * m.accuracy);
  return 0;
}
