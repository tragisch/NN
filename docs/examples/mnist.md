# Example: MNIST

Source: `examples/mnist_api/main.c`. The program trains a 784-128-10 network on
10 000 training images, validates on a held-out tenth of them every epoch and
reports accuracy on 2 000 test images. The user code is about fifteen lines;
the rest loads the IDX files with the helpers in `examples/mnist/mnist_data.c`
(not part of the library).

Run it from the repository root:

```text
bazel run //examples/mnist_api:mnist_api
```

```c
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
```

Typical output:

```text
epoch  1/10  loss 0.3817  val 0.3795
...
epoch 10/10  loss 0.0650  val 0.1694
test accuracy 92.80%
```

Notes:

- `load_mnist_images` returns a `FloatMatrix` (mMatrix type); `view` wraps its
  buffer in an `NmTensor` without copying. In your own program you can pass any
  `float` array.
- Labels are one-hot rows (`load_mnist_labels_onehot`), the output layer is
  Softmax, and the default loss with targets is cross-entropy.
- The same task with the legacy API is `examples/mnist/main.c`
  ([legacy page](../legacy/mnist.md)); it also supports a CNN and
  benchmarks against PyTorch.
- Add `fit.optimizer = NM_OPT_ADAM; fit.learning_rate = 0.001f;` for faster
  convergence than the plain SGD default.
