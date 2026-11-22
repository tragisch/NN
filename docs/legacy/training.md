# API: Training & Inference

!!! warning "Legacy API"
    This page describes the legacy `nm.h` API, which stays available but is not
    the recommended entry point. New programs should use `nm_api.h`; see
    [API Overview](../api/overview.md).

This section covers the unified `nm_*` entry points for fitting, prediction, evaluation, and a few lower-level math helpers that support custom workflows.

## Example: fit + predict + accuracy

```c
// X_train: [num_samples x input_dim]
// Y_train: [num_samples x num_classes] (one-hot)
NmTrainConfig train_cfg = nm_train_defaults();
train_cfg.batch_size = 128;
train_cfg.learning_rate = 0.02f;
train_cfg.epochs = 10;
train_cfg.verbose = true;

NmStatus s = nm_fit(model, X_train, Y_train, &train_cfg);
if (s != NM_STATUS_OK) {
	// handle training error
}

FloatMatrix *y_pred = NULL;
s = nm_predict(model, X_test, &y_pred);
if (s == NM_STATUS_OK) {
	float acc = 0.0f;
	s = nm_accuracy(y_pred, Y_test, &acc);
	sm_destroy(y_pred);
}
```

## Training entry points

- `nm_fit_epoch`
- `nm_fit`

### When to use which one

- `nm_fit_epoch`: useful for your own outer loop, custom logging, early stopping, or curriculum logic.
- `nm_fit`: the simple multi-epoch training path.

### Usage notes

- `X` is expected as `[num_samples x input_dim]`.
- `Y` is expected as `[num_samples x output_dim]`.
- Hyperparameters live in `NmTrainConfig`.
- For classification, targets are commonly one-hot encoded.

## Inference entry points

- `nm_predict`
- `nm_predict_into`
- `nm_evaluate`
- `nm_save_inference`

### Behavior

- `nm_predict` allocates a fresh output matrix and returns it through `out_prediction`.
- `nm_predict_into` writes into a caller-provided output matrix with matching shape.
	This is the lowest-allocation public inference path and reuses the model's
	prepared tensor workspace after warm-up.
- `nm_evaluate` computes an evaluation metric directly from `X` and `Y`.

Internally, inference now runs through the compiled execution plan and a
dedicated prepared workspace that reuses tensor-pool allocations across calls.
Dense-only models still use specialized GEMM fastpaths, but those fastpaths are
selected as plan strategies instead of public prediction branches. The first
prediction for a new batch shape may populate the workspace; repeated calls with
similar shapes avoid repeated heap churn in the hot path.

## Inference-only exports

Use `nm_save_inference(model, path)` when you want to persist only the parts of
the model needed for inference. The resulting file can still be loaded with
`nm_load(...)`, but `nm_model_is_inference_only(...)` will return `true` and
training-oriented setters such as `nm_set_autograd_mode(...)` will reject it.

## Example: one-epoch control loop

```c
for (size_t epoch = 0; epoch < 5; ++epoch) {
	NmTrainConfig cfg = nm_train_defaults();
	cfg.batch_size = 64;
	cfg.learning_rate = 0.01f;
	s = nm_fit_epoch(model, X_train, Y_train, &cfg);
}
```

## Forward math helpers

- `nm_linear`
- `nm_linear_into`

These functions implement the common linear transform `input * weights + bias` and are useful in custom experiments, debug tooling, or lower-level layer work.

## Activation and loss helpers

The public API still exposes reusable helpers such as:

- `nm_relu`, `nm_sigmoid`, `nm_tanh`, `nm_softmax`
- loss helpers like `nm_loss_mse` and `nm_loss_cross_entropy`
- reduction helpers such as `nm_sum_rows` and `nm_argmax_rows`

That keeps one model API while still offering building blocks for more advanced usage. One API, many toys.
