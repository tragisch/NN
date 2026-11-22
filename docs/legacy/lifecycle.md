# API: Lifecycle & Configuration

!!! warning "Legacy API"
    This page describes the legacy `nm.h` API, which stays available but is not
    the recommended entry point. New programs should use `nm_api.h`; see
    [API Overview](../api/overview.md).

This section explains model ownership, setup order, configuration defaults, and runtime knobs for the unified `nm_*` API.

## Example: create, compile, destroy

```c
#include "nm.h"

NmShape input = {.ndim = 1, .dims = {784}};
NmModel *model = nm_create(input);
if (!model) {
	return;
}

NmDenseConfig hidden = {
	.units = 128,
	.activation = NM_ACT_RELU,
};
NmDenseConfig output = {
	.units = 10,
	.activation = NM_ACT_SOFTMAX,
};

NmStatus s = nm_add_dense(model, &hidden);
if (s == NM_STATUS_OK) {
	s = nm_add_dense(model, &output);
}

NmCompileConfig cfg = nm_compile_defaults();
cfg.loss_type = NM_LOSS_CROSS_ENTROPY;

if (s == NM_STATUS_OK) {
	s = nm_compile(model, &cfg);
}

if (s != NM_STATUS_OK) {
	nm_free(model);
	return;
}

nm_free(model);
```

## Default config constructors

### `nm_compile_defaults`

Returns a safe baseline `NmCompileConfig`. For most models you only need to set `loss_type` before calling `nm_compile(...)`.

### `nm_train_defaults`

Returns baseline `NmTrainConfig` for training loops.

### `nm_optimizer_defaults`

Returns a baseline `NmOptimizerConfig` for the model optimizer.

Use these helpers whenever you only want to override one or two fields instead of filling everything manually.

## Model lifetime

### `nm_create`

- Allocates a new model on the heap.
- Requires a valid `NmShape` for the model input.
- Returns `NULL` on allocation or validation failure.

### `nm_free`

- Releases all model-owned resources.
- Safe to call with `NULL`.

## Builder phase

Add layers in order after `nm_create` and before `nm_compile`.

Available builder calls include:

- `nm_add_dense`
- `nm_add_conv2d`
- `nm_add_maxpool2d`
- `nm_add_avgpool2d`
- `nm_add_batchnorm2d`
- `nm_add_flatten`

This gives you one clear topology-building phase instead of juggling a separate network config object.

## Compile phase

### `nm_compile`

Applies training behavior and validates that the model is internally consistent.

Important:

- Pass `NULL` to use `nm_compile_defaults()` unchanged.
- Cross-entropy + softmax output automatically uses the fused shortcut internally.
- Compile builds the internal inference execution plan and resets the prepared
	inference workspace so the next predict path can rebuild it for the current
	topology.
- Invalid combinations still return an error status instead of silently proceeding.

## Optional runtime controls

These APIs remain within the same public model interface:

- `nm_set_optimizer`
- `nm_set_backend`
- `nm_model_backend`
- `nm_model_backend_capabilities`
- `nm_set_tensor_layout` / `nm_tensor_layout`
- `nm_set_autograd_mode` / `nm_autograd_mode`
- `nm_set_batch_parallel` / `nm_batch_parallel`
- `nm_last_graph_nodes`
- `nm_model_is_inference_only`

`nm` now keeps training state and inference runtime state separate internally:

- optimizer/autograd/batch-parallel/tape state live on the training side
- the compiled execution plan, prepared tensor-pool reuse and inference
  workspace sizing live on the runtime side

This keeps the public API compact while allowing dense-only fastpaths and
generic spatial layers to use one internal inference runner.

So the lifecycle is simple:

1. create
2. add layers
3. compile
4. optionally tune runtime behavior
5. fit / predict / save
6. free
