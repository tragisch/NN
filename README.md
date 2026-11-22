# NN Library

A compact C neural-network library with explicit frontend, local C-engine and
GraphIR components.

## Quick Start

Requires: macOS on Apple Silicon, [Bazel](https://bazel.build) 10 (see `.bazelversion`).

```bash
git clone https://github.com/tragisch/14_NN.git
cd 14_NN
bazel build //...              # build everything
bazel test //app/nm_lib/...    # run tests
bazel run //examples/xor:xor   # run the XOR example
```

See [docs/getting-started/installation.md](docs/getting-started/installation.md) for details.

## Public Build Targets

- `//app/nm_lib:nm_api` — the user API (`nm_api.h`): describe, build, train,
  predict and save a network with one handle. Start here.
- `//app/nm_lib:nm_frontend` — model topology and initial weights. Finalizing
  a frontend model builds a GraphIR-shaped graph that owns its weights.
- `//app/nm_lib:nm_engine_c` — local training and inference on such a graph.
- `//app/nm_lib:nm_export_graphir` — GraphIR v4 JSON export of a model or graph.
- `//app/nm_lib:nm_connector` — weight sets, zero-copy binding and publisher /
  subscriber control on top of pluggable transports (in-process queue, files).
- `//app/nm_lib:nm_import_graphir` — GraphIR v3/v4 import, including
  `.girw` weight packages (`nm_girw.h`), producing a graph the engine runs and
  trains directly.

The former unified `nm.h` workflow is a legacy entry point that shares the
same trainer. Build and finalize a frontend model once, then choose a local C
engine, GraphIR export, or both. A graph imported from GraphIR needs no
frontend model at all.

## Current Scope

Implemented and usable today:

- Dense, `Conv2D`, `MaxPool2D` and `Flatten` models with ReLU, Sigmoid, Tanh
  and Softmax activations
- Inference on imported GraphIR graphs that are not layer chains (residual
  connections, attention) through a graph interpreter: broadcasting
  Add/Sub/Mul/Div, batched MatMul, Transpose, Reshape, LayerNorm and GELU up
  to rank 4; these graphs cannot be trained
- Explicit frontend finalization and C-engine preparation
- Local C training, inference and evaluation through
  `nm_engine_c_prepare` or `nm_engine_c_prepare_graph`
- GraphIR v4 export (opset 1 for Dense models, opset 2 with CNN operations)
  and import of Dense, CNN and activation chains with inline or `.girw`
  weights, with XXH64 checksum verification for `.girw` version 2
- Weight transfer between components: a trained graph publishes its weights
  as a versioned, checksummed `.girw`-based message; a receiver with the same
  structure binds them without copying between two predicts, keeps computing
  when the publisher disappears and follows a restarted publisher
- Legacy `.nms*` import of topology and weights through
  `nm_frontend_import_legacy`; training and runtime state are intentionally
  discarded

GraphIR export rejects quantization and non-float32 weights. `AvgPool2D` and
`BatchNorm2D` are not exposed by the frontend API.

## Repository Layout

The repository separates library code from runnable samples:

- `app/nm_lib/` — the reusable library, public headers, sources, and tests
- `examples/` — runnable sample programs, including GraphIR export
- `docs/` — documentation and guides
- `share/` — benchmarks and supporting scripts

## User API

```c
#include "nm_api.h"

NmSpec *spec = nm_spec_new((NmShape){.ndim = 1, .dims = {784}});
nm_spec_dense(spec, 128, NM_ACT_RELU);
nm_spec_dense(spec, 10, NM_ACT_SOFTMAX);

NmBuildOptions build = nm_build_options_default();
build.seed = 42;
NmNetwork *net = nm_network_build(spec, &build);   /* NULL: see nm_last_error() */

NmFitOptions fit = nm_fit_options_default();
fit.epochs = 10;
nm_network_fit(net, x_train, &y_train, &fit, on_progress, NULL);

NmMetrics m;
nm_network_evaluate(net, x_test, y_test, NM_LOSS_CROSS_ENTROPY, &m);
nm_network_save(net, "mnist.nmmodel");
```

Pass `NULL` targets to `nm_network_fit` to train an autoencoder.
`nm_network_summary` prints layers, output shapes and parameter counts. A
complete MNIST program is in [examples/mnist_api](examples/mnist_api/main.c), a
self-contained regression program in [docs/examples/regression.md](docs/examples/regression.md). The handle is
called `NmNetwork` because `NmModel` belongs to the legacy `nm.h`. The header
imports into Swift through `include/module.modulemap`. The design notes are in
[docs/api_m1_entwurf.md](docs/api_m1_entwurf.md).

For regression set `fit.loss = NM_LOSS_MSE` explicitly: with `NM_LOSS_NONE` and
targets the library trains with cross-entropy.

### Use without Bazel

Build the self-contained archive once and link it from any build system. The
mMatrix header directory is needed as well, because `nm_types.h` includes
`sm.h`:

```text
bazel build //app/nm_lib/ffi:nm_api_static

clang -O2 -std=c11 my_program.c \
  -Iapp/nm_lib/include -I/path/to/mMatrix/app/matrix/include \
  bazel-bin/app/nm_lib/ffi/libnm_api_static.a \
  -framework Accelerate -framework Metal -framework MetalPerformanceShaders \
  -framework MetalPerformanceShadersGraph -framework Foundation \
  -o my_program
```

Details: [docs/getting-started/installation.md](docs/getting-started/installation.md).

### Known limits

No CSV or data loader, no gradient with respect to the input, no custom loss,
no single training step and no direct weight access. Optimization on a trained
surrogate works through `predict` only
([docs/api/optimization.md](docs/api/optimization.md)). There is no API call
for the thread count; `OMP_NUM_THREADS` is honoured.

## Quick Example (frontend and engine)

```c
#include "nm_engine_c.h"
#include "nm_export_graphir.h"

NmShape input = {.ndim = 1, .dims = {4}};
NmFrontendModel *model = nm_frontend_create(input);
if (!model) {
  return 1;
}

NmStatus s = nm_frontend_add_dense(model, &(NmDenseConfig){
    .units = 8,
    .activation = NM_ACT_RELU,
});
if (s == NM_STATUS_OK) {
  s = nm_frontend_add_dense(model, &(NmDenseConfig){
      .units = 3,
      .activation = NM_ACT_SOFTMAX,
  });
}
if (s == NM_STATUS_OK) {
  s = nm_frontend_finalize(model);
}
if (s == NM_STATUS_OK) {
  NmTrainConfig train_config = {
      .batch_size = 32,
      .learning_rate = 0.01f,
      .epochs = 1,
      .verbose = false,
  };
  NmEngineC *engine = nm_engine_c_create(NULL);
  s = engine ? nm_engine_c_prepare(engine, model) : NM_STATUS_ALLOC_FAILED;
  if (s == NM_STATUS_OK) {
    s = nm_engine_c_fit(engine, model, X, Y, &train_config);
  }
  nm_engine_c_free(engine);
}
if (s == NM_STATUS_OK) {
  s = nm_export_graphir_json(
      model, &(NmGraphIrExportOptions){.batch_size = X->rows}, "model.json");
}
nm_frontend_free(model);
```

Run the complete XOR example with local training, prediction and GraphIR
export:

```text
bazel run //examples/graphir_export:graphir_export
```

It writes `xor.graphir.json`. Validate and generate dependency-free Rust with
a local GraphIR checkout:

```text
GRAPHIR_ROOT=/path/to/GraphIR tools/verify_graphir_export.sh
```

## Entry Points

- Documentation start: [docs/index.md](docs/index.md)
- Installation (Bazel and static archive): [docs/getting-started/installation.md](docs/getting-started/installation.md)
- First walkthrough (classification and regression): [docs/getting-started/first-network.md](docs/getting-started/first-network.md)
- Usage guide: [docs/getting-started/library-usage.md](docs/getting-started/library-usage.md)
- API map: [docs/api/overview.md](docs/api/overview.md)
- Optimization with a trained network: [docs/api/optimization.md](docs/api/optimization.md)
- Examples: [MNIST](docs/examples/mnist.md), [regression](docs/examples/regression.md)
- Legacy `nm.h` API: [docs/legacy/index.md](docs/legacy/index.md)
- Design notes: [app/nm_lib/docs/api_m1_entwurf.md](app/nm_lib/docs/api_m1_entwurf.md), [docs/plans/2026-08-22-nm-frontend-engine-graphir.md](docs/plans/2026-08-22-nm-frontend-engine-graphir.md)
- Runnable GraphIR example: [examples/graphir_export/README.md](examples/graphir_export/README.md)

Build the documentation site with `bazel build //:mkdocs`.
