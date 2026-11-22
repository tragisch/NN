# Your First Network

Two complete programs: classification and regression. Both use only
`nm_api.h` and check every status, so they are also a template for error
handling. Build instructions are in [Installation](installation.md).

## Regression: learn a curve

```c
#include "nm_api.h"
#include <math.h>
#include <stdio.h>

#define N 256

int main(void) {
  static float x[N], y[N];
  for (int i = 0; i < N; ++i) {
    x[i] = -2.0f + 4.0f * (float)i / (N - 1);
    y[i] = sinf(3.0f * x[i]) + 0.5f * x[i];
  }

  NmSpec *spec = nm_spec_new((NmShape){.ndim = 1, .dims = {1}});
  nm_spec_dense(spec, 32, NM_ACT_TANH);
  nm_spec_dense(spec, 32, NM_ACT_TANH);
  nm_spec_dense(spec, 1, NM_ACT_NONE);       /* linear output for regression */

  NmBuildOptions build = nm_build_options_default();
  build.seed = 42;
  NmNetwork *net = nm_network_build(spec, &build);
  nm_spec_free(spec);                        /* the network keeps no reference */
  if (!net) {
    fprintf(stderr, "build: %s\n", nm_last_error());
    return 1;
  }

  NmTensor inputs = nm_tensor_2d(x, N, 1);
  NmTensor targets = nm_tensor_2d(y, N, 1);

  NmFitOptions fit = nm_fit_options_default();
  fit.loss = NM_LOSS_MSE;                    /* say so explicitly for regression */
  fit.optimizer = NM_OPT_ADAM;
  fit.learning_rate = 0.01f;
  fit.epochs = 200;
  if (nm_network_fit(net, inputs, &targets, &fit, NULL, NULL) != NM_STATUS_OK) {
    fprintf(stderr, "fit: %s\n", nm_last_error());
    nm_network_free(net);
    return 1;
  }

  NmMetrics m;
  nm_network_evaluate(net, inputs, targets, NM_LOSS_MSE, &m);
  printf("MSE %.5f\n", m.loss);

  NmTensor out;
  float query[1] = {0.5f};
  if (nm_network_predict(net, nm_tensor_2d(query, 1, 1), &out) == NM_STATUS_OK) {
    printf("f(0.5) = %.3f (exact %.3f)\n", out.data[0],
           sinf(1.5f) + 0.25f);
    nm_tensor_free(&out);                    /* predict allocates, you free */
  }
  nm_network_free(net);
  return 0;
}
```

With these settings the training error ends around `0.001`; the printed value for `f(0.5)` is close to the exact `1.247`.

## Classification: three classes

Targets are one-hot rows, the output layer is Softmax and the loss is
cross-entropy (which is also what `NM_LOSS_NONE` selects when you pass
targets).

```c
#include "nm_api.h"
#include <stdio.h>

static int on_epoch(const NmProgress *p, void *user) {
  (void)user;
  if (p->epoch % 50 == 0)
    printf("epoch %3zu/%zu  loss %.4f\n", p->epoch, p->epochs, p->loss);
  return 0;                                  /* non-zero would stop training */
}

/* x: n rows of 4 features, y: n rows of 3 one-hot values */
int train_classifier(float *x, float *y, size_t n) {
  NmSpec *spec = nm_spec_new((NmShape){.ndim = 1, .dims = {4}});
  nm_spec_dense(spec, 8, NM_ACT_RELU);
  nm_spec_dense(spec, 3, NM_ACT_SOFTMAX);
  NmNetwork *net = nm_network_build(spec, NULL);
  nm_spec_free(spec);
  if (!net)
    return 1;

  NmTensor inputs = nm_tensor_2d(x, n, 4);
  NmTensor targets = nm_tensor_2d(y, n, 3);

  NmFitOptions fit = nm_fit_options_default();
  fit.loss = NM_LOSS_CROSS_ENTROPY;
  fit.epochs = 200;
  fit.batch_size = 16;
  fit.validation_split = 0.2f;               /* last 20 % of the rows */
  if (nm_network_fit(net, inputs, &targets, &fit, on_epoch, NULL) != NM_STATUS_OK) {
    fprintf(stderr, "fit: %s\n", nm_last_error());
    nm_network_free(net);
    return 1;
  }

  NmMetrics m;
  nm_network_evaluate(net, inputs, targets, NM_LOSS_CROSS_ENTROPY, &m);
  printf("loss %.4f  accuracy %.1f%%\n", m.loss, 100.0 * m.accuracy);

  nm_network_save(net, "classifier.nmmodel");
  nm_network_free(net);
  return 0;
}
```

## The mental model

1. **Describe** with an `NmSpec`: input shape, then layers in order.
2. **Build** an `NmNetwork`: weights are drawn, the engine is prepared. One
   spec can be built many times (ensembles, hyper-parameter runs).
3. **Fit** with an `NmFitOptions` struct. Start from
   `nm_fit_options_default()` and change only what you need.
4. **Predict / evaluate** on new rows.
5. **Save** to a directory; **load** it back in another program.
6. **Free** the network. Always check the returned `NmStatus` or `NULL`, and
   print `nm_last_error()` on failure.

## Where to go next

- [Library Usage](library-usage.md): data layout, defaults and pitfalls
- [Optimization with a trained network](../api/optimization.md)
- `examples/xor`, `examples/iris`, `examples/boston` (legacy API, full
  programs with data)
