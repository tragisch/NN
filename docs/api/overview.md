# API Overview

The public API is `app/nm_lib/include/nm_api.h`. Everything a program needs to
describe, build, train, run and save a network is in that one header. The other
headers are lower-level building blocks that `nm_api` itself is made of; most
programs never include them.

## `nm_api.h` at a glance

| Group | Functions |
| --- | --- |
| Data | `NmTensor`, `nm_tensor_2d`, `nm_tensor_numel`, `nm_tensor_free` |
| Describe | `nm_spec_new`, `nm_spec_dense`, `nm_spec_conv2d`, `nm_spec_maxpool2d`, `nm_spec_flatten`, `nm_spec_free` |
| Build | `nm_build_options_default`, `nm_network_build`, `nm_network_free` |
| Train | `nm_fit_options_default`, `nm_network_fit`, `NmProgressFn` |
| Use | `nm_network_predict`, `nm_network_predict_inputs`, `nm_network_predict_batches`, `nm_network_evaluate`, `nm_network_input_count`, `nm_network_input_info` |
| Inspect | `nm_network_parameter_count`, `nm_network_summary` |
| Persist | `nm_network_save`, `nm_network_load`, `nm_network_import_graphir`, `nm_network_export_graphir` |
| Errors | `NmStatus`, `nm_last_error` |

Handles are opaque (`NmSpec`, `NmNetwork`). Types are plain C, so the header
imports into Swift through `include/module.modulemap` and is annotated with
`_Nonnull` / `_Nullable` under clang.

## Typical call sequence

1. `nm_spec_new` and `nm_spec_*` to describe the layers
2. `nm_network_build` (then `nm_spec_free`, the spec can also be built again)
3. `nm_network_fit`
4. `nm_network_predict` / `nm_network_evaluate`
5. `nm_network_save`, later `nm_network_load`
6. `nm_network_free`

## Types

- `NmStatus`: `NM_STATUS_OK` or an error class.
- `NmShape`: the shape of one example, `ndim` 1 to 3.
- `NmActivation`: `NONE`, `RELU`, `SIGMOID`, `TANH`, `SOFTMAX`.
- `NmLoss`: `NONE` (decide from the targets), `MSE`, `CROSS_ENTROPY`.
- `NmOptimizerType`: `SGD`, `MOMENTUM`, `RMSPROP`, `ADAM`.
- `NmBackendType`: `AUTO`, `DEFAULT` (both select the same kernels today).
- `NmBuildOptions`, `NmFitOptions`, `NmProgress`, `NmMetrics`: small structs
  with `*_default()` constructors where they carry options.
- `NmInput`: one named, typed model input (`float32`, `int64`, `bool`).

## Other headers

| Header | Purpose | Page |
| --- | --- | --- |
| `nm_frontend.h` | build a model layer by layer, finalize into a graph | [Components](components.md) |
| `nm_engine_c.h` | train and run graphs, bind shared weights | [Components](components.md) |
| `nm_import_graphir.h`, `nm_export_graphir.h` | GraphIR v3/v4 JSON | [Serialization](serialization.md) |
| `nm_girw.h` | read and write `.girw` weight packages | [Serialization](serialization.md) |
| `nm_weightset.h`, `nm_connector.h` | weights as versioned messages, publisher and subscriber | [Components](components.md) |
| `nm_capabilities.h` | JSON profile of what the engine can run | [Components](components.md) |
| `nm.h` | older `NmModel` API on `FloatMatrix` | [Legacy](../legacy/index.md) |

## Design notes

`nm_api` is a thin layer over the frontend, graph and engine; there is no
second implementation. The design notes (in German) are in
`app/nm_lib/docs/api_m1_entwurf.md`, the data type and backend roadmap in
`m2_datentypen_backend.md`. The export-oriented architecture criteria are in
`design-principles.md`.
