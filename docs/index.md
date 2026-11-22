# NN Library

`nm_lib` is a small neural-network library in C. You describe a network, train
it on your own data, run it and save it, all through one header, `nm_api.h`,
with opaque handles and plain C types.

```c
#include "nm_api.h"

NmSpec *spec = nm_spec_new((NmShape){.ndim = 1, .dims = {784}});
nm_spec_dense(spec, 128, NM_ACT_RELU);
nm_spec_dense(spec, 10, NM_ACT_SOFTMAX);
NmNetwork *net = nm_network_build(spec, NULL);
nm_network_fit(net, x_train, &y_train, NULL, NULL, NULL);
```

## What you can do today

- **Train** dense and convolutional chains: `Dense`, `Conv2D`, `MaxPool2D`,
  `Flatten` with ReLU, Sigmoid, Tanh and Softmax. Losses: mean squared error
  and cross-entropy. Optimizers: SGD, Momentum, RMSProp, Adam. Autoencoders
  train by passing no targets.
- **Predict** with models of your own, or with models imported from
  [GraphIR](api/serialization.md), including graphs that are not layer chains
  (residual connections, attention) and models with several named inputs of
  type float32, int64 or bool. Imported graphs of that kind are inference-only.
- **Save and exchange**: a directory with a GraphIR v4 JSON file and a
  checksummed `.girw` weight package, or one self-contained JSON file.
- **Move weights between processes** with versioned weight sets, zero-copy
  binding and a publisher/subscriber connector
  ([Components](api/components.md)).

## What it does not do (yet)

- No gradient with respect to the input, no custom loss functions, no single
  training step and no direct access to the weights. Optimization on a
  trained surrogate therefore works through `predict` only
  ([Optimization](api/optimization.md)).
- No loader for CSV or other data formats. Data goes in as row-major `float`
  arrays ([Library Usage](getting-started/library-usage.md)).
- `AvgPool2D` and `BatchNorm2D` are not exposed. Recurrent layers are not
  implemented.
- Training runs in float32. Float16 and bfloat16 graphs are accepted by
  promotion for inference.
- Built for macOS on Apple Silicon (Accelerate, Metal Performance Shaders).

## Where to start

1. [Installation](getting-started/installation.md): build with Bazel or link
   the static archive from any build system.
2. [First Network](getting-started/first-network.md): classification and
   regression in about twenty lines each.
3. [Library Usage](getting-started/library-usage.md): data layout, options,
   errors, threads, reproducibility.
4. [API Overview](api/overview.md): every header and what it is for.
5. [Examples](examples/mnist.md): MNIST and a regression program.

The older `nm.h` API still works and is documented under
[Legacy](legacy/index.md).
