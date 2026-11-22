# Example: Regression on Noisy Data

A complete program for the typical "fit a function from samples" task: build
data, normalize it, hold out test rows, train with a validation split and
early stopping, measure the error in the original units and save the model.

It is self-contained (no data files) and links against the library as
described in [Installation](../getting-started/installation.md).

```c
/* Regression with noisy data: shuffle, normalize, hold out a test set,
 * train with early stopping, report MAE on the test rows, save the model. */
#include "nm_api.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 600
#define TEST 100
#define F 2

static float frand(void) { return (float)rand() / (float)RAND_MAX; }

typedef struct {
  float best;
  size_t bad;
} Stop;

static int on_epoch(const NmProgress *p, void *user) {
  Stop *s = user;
  if (p->val_loss < s->best - 1e-5f) {
    s->best = p->val_loss;
    s->bad = 0;
  } else if (++s->bad >= 15) {
    printf("early stop at epoch %zu (best val %.5f)\n", p->epoch, s->best);
    return 1;
  }
  return 0;
}

int main(void) {
  static float x[N * F], y[N];
  srand(7);
  for (int i = 0; i < N; ++i) {          /* y = 3a - 2b^2 + noise, already shuffled */
    float a = 10.0f * frand(), b = 4.0f * frand() - 2.0f;
    x[i * F] = a;
    x[i * F + 1] = b;
    y[i] = 3.0f * a - 2.0f * b * b + 0.1f * (frand() - 0.5f);
  }

  /* z-score every feature and the target with the training statistics */
  const size_t train = N - TEST;
  float mean[F + 1] = {0}, sd[F + 1] = {0};
  for (size_t i = 0; i < train; ++i) {
    for (int k = 0; k < F; ++k) mean[k] += x[i * F + k] / (float)train;
    mean[F] += y[i] / (float)train;
  }
  for (size_t i = 0; i < train; ++i) {
    for (int k = 0; k < F; ++k) sd[k] += powf(x[i * F + k] - mean[k], 2) / (float)train;
    sd[F] += powf(y[i] - mean[F], 2) / (float)train;
  }
  for (int k = 0; k <= F; ++k) sd[k] = sqrtf(sd[k]);
  for (size_t i = 0; i < N; ++i) {
    for (int k = 0; k < F; ++k) x[i * F + k] = (x[i * F + k] - mean[k]) / sd[k];
    y[i] = (y[i] - mean[F]) / sd[F];
  }

  NmSpec *spec = nm_spec_new((NmShape){.ndim = 1, .dims = {F}});
  nm_spec_dense(spec, 32, NM_ACT_TANH);
  nm_spec_dense(spec, 16, NM_ACT_TANH);
  nm_spec_dense(spec, 1, NM_ACT_NONE);
  NmBuildOptions build = nm_build_options_default();
  build.seed = 1;
  NmNetwork *net = nm_network_build(spec, &build);
  nm_spec_free(spec);
  if (!net) {
    fprintf(stderr, "build: %s\n", nm_last_error());
    return 1;
  }

  NmTensor tx = nm_tensor_2d(x, train, F), ty = nm_tensor_2d(y, train, 1);
  NmFitOptions fit = nm_fit_options_default();
  fit.loss = NM_LOSS_MSE;
  fit.optimizer = NM_OPT_ADAM;
  fit.learning_rate = 0.005f;
  fit.epochs = 300;
  fit.batch_size = 32;
  fit.validation_split = 0.15f;            /* data is already in random order */
  Stop stop = {.best = INFINITY};
  if (nm_network_fit(net, tx, &ty, &fit, on_epoch, &stop) != NM_STATUS_OK) {
    fprintf(stderr, "fit: %s\n", nm_last_error());
    nm_network_free(net);
    return 1;
  }

  /* held-out test rows: MAE in the original units */
  NmTensor out;
  if (nm_network_predict(net, nm_tensor_2d(x + train * F, TEST, F), &out) != NM_STATUS_OK) {
    fprintf(stderr, "predict: %s\n", nm_last_error());
    nm_network_free(net);
    return 1;
  }
  double mae = 0;
  for (size_t i = 0; i < TEST; ++i)
    mae += fabs((double)(out.data[i] - y[train + i]) * sd[F]);
  printf("test MAE %.3f (target std %.2f)\n", mae / TEST, sd[F]);
  nm_tensor_free(&out);

  if (nm_network_save(net, "regression_model") != NM_STATUS_OK)
    fprintf(stderr, "save: %s\n", nm_last_error());
  nm_network_free(net);
  return 0;
}
```

Output:

```text
early stop at epoch 94 (best val 0.00029)
test MAE 0.174 (target std 9.39)
```

What to take from it:

- **Normalize** features and targets with statistics from the training rows
  only, and undo the scaling on the output (`* sd[F]` above).
- **`validation_split` uses the last rows**, so shuffle your data first
  (here it is generated in random order).
- **Early stopping** is done in the progress callback by returning non-zero.
  The library does not restore the best weights; with a small patience the
  final weights are close to the best ones. To keep the best exactly, train in
  short `fit` calls and save after the best one. Do not call other functions on
  the same network from inside the callback: it runs while the network's lock
  is held.
- Set `fit.loss = NM_LOSS_MSE` explicitly.
- Save with `nm_network_save` and reload with `nm_network_load`; the loaded
  network predicts any number of rows. Remember to apply the same
  normalization statistics to new data. They are not stored in the model.
