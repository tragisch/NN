# Optimization with a Trained Network

`nm_lib` trains and runs networks; it is not an optimizer toolkit. A common
pattern still works well: train a network as a **surrogate** of an expensive
function, then search for its minimum or maximum by evaluating the network.

## What the API gives you

- `nm_network_predict` evaluates many points in one call (a batch). This is
  the efficient way to scan a grid or a random population.
- `nm_network_fit` can be called again with new data (active learning, a
  refinement loop around the real function).
- Weights and the network structure can be saved and reloaded
  ([Serialization](serialization.md)).

## What it does not give you

These are the gaps you will hit when you move from classification to
optimization:

| Missing | Consequence |
| --- | --- |
| Gradient of the output with respect to the **input** | use finite differences, or a gradient-free search |
| Custom loss function (weighted MSE, Huber, physics residuals) | only `MSE` and `CROSS_ENTROPY`; reshape or weight your targets instead |
| One training step on one batch | `fit` runs whole epochs over a fixed dataset |
| Direct access to the weights as a vector | cannot drive CMA-ES, L-BFGS and similar from outside |
| Weight decay, gradient clipping, learning-rate schedules | optimizer settings are `optimizer` and `learning_rate` only |
| Regression metrics (MAE, R²) | compute them from `predict` |

They are known, and they are the planned next steps for the library.

## Example: minimize a learned curve

Train on samples of `f(x) = sin(3x) + 0.5x`, scan a grid in one batch, then
refine with central differences:

```c
#include "nm_api.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Surrogate f(x) = sin(3x) + 0.5x on [-2, 2]; find its minimum. */
static float surrogate(const NmNetwork *net, float x) {
  NmTensor out;
  float in[1] = {x};
  if (nm_network_predict(net, nm_tensor_2d(in, 1, 1), &out) != NM_STATUS_OK)
    return NAN;
  float v = out.data[0];
  nm_tensor_free(&out);
  return v;
}

int main(void) {
  enum { N = 256 };
  static float x[N], y[N];
  for (int i = 0; i < N; ++i) {
    x[i] = -2.0f + 4.0f * (float)i / (N - 1);
    y[i] = sinf(3.0f * x[i]) + 0.5f * x[i];
  }
  NmSpec *spec = nm_spec_new((NmShape){.ndim = 1, .dims = {1}});
  nm_spec_dense(spec, 32, NM_ACT_TANH);
  nm_spec_dense(spec, 32, NM_ACT_TANH);
  nm_spec_dense(spec, 1, NM_ACT_NONE);
  NmBuildOptions b = nm_build_options_default();
  b.seed = 42;
  NmNetwork *net = nm_network_build(spec, &b);
  nm_spec_free(spec);
  if (!net) return 1;
  NmTensor xt = nm_tensor_2d(x, N, 1), yt = nm_tensor_2d(y, N, 1);
  NmFitOptions fit = nm_fit_options_default();
  fit.loss = NM_LOSS_MSE;
  fit.optimizer = NM_OPT_ADAM;
  fit.learning_rate = 0.01f;
  fit.epochs = 200;
  if (nm_network_fit(net, xt, &yt, &fit, NULL, NULL) != NM_STATUS_OK) return 1;

  /* 1. Coarse grid in ONE predict call (batch), pick the best start. */
  enum { G = 401 };
  static float grid[G];
  for (int i = 0; i < G; ++i) grid[i] = -2.0f + 0.01f * (float)i;
  NmTensor out;
  if (nm_network_predict(net, nm_tensor_2d(grid, G, 1), &out) != NM_STATUS_OK) return 1;
  int best = 0;
  for (int i = 1; i < G; ++i)
    if (out.data[i] < out.data[best]) best = i;
  float xm = grid[best];
  nm_tensor_free(&out);
  printf("grid:   x = %.3f  f = %.3f\n", xm, surrogate(net, xm));

  /* 2. Refine with central differences and plain gradient descent. */
  const float h = 1e-2f, step = 0.05f;
  for (int it = 0; it < 100; ++it) {
    float g = (surrogate(net, xm + h) - surrogate(net, xm - h)) / (2.0f * h);
    xm -= step * g;
    if (xm < -2.0f) xm = -2.0f;
    if (xm > 2.0f) xm = 2.0f;
  }
  printf("refined: x = %.3f  f = %.3f  (true f = %.3f)\n", xm,
         surrogate(net, xm), sinf(3.0f * xm) + 0.5f * xm);
  nm_network_free(net);
  return 0;
}
```

Output:

```text
grid:   x = -0.560  f = -1.274
refined: x = -0.558  f = -1.274  (true f = -1.274)
```

## Practical notes

- **Normalize inputs and targets** to a similar range before training. Raw
  values with very different scales make small networks train badly.
- **A surrogate is only trustworthy where it saw data.** Clamp the search to
  the training range, as above, or add a margin check.
- **Finite differences cost two predicts per dimension.** For more than a
  handful of dimensions, prefer a population-based search that evaluates a
  whole batch per generation.
- **Use `tanh` for smooth functions.** ReLU networks produce piecewise-linear
  surrogates with many flat or kinked regions, which confuses finite
  differences.
- For `predict` in a tight loop, keep one network per thread
  ([Library Usage](../getting-started/library-usage.md#threads)).
