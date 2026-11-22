# API: Metrics & Utilities

!!! warning "Legacy API"
    This page describes the legacy `nm.h` API, which stays available but is not
    the recommended entry point. New programs should use `nm_api.h`; see
    [API Overview](../api/overview.md).

## `nm_accuracy`

Purpose:

- Classification metric comparing predicted and target one-hot matrices.

Benefit:

- Immediate quality signal after training or inference runs.

## `nm_argmax_rows`

Purpose:

- Convert per-class probabilities/logits into label indices.

Benefit:

- Reusable utility for metrics and downstream post-processing.

## `nm_sum_rows`

Purpose:

- Sum each column across rows.

Benefit:

- Used in bias gradient calculation; also useful as a generic reduction helper.

## `nm_add_rowwise`

Purpose:

- Add row vector bias to every row.

Benefit:

- Key primitive for dense-layer forward pass.

## Runtime/utility functions

### `nm_backend`

Returns the active compute backend name (for diagnostics/logging).

### `nm_backend_capabilities` / `nm_model_backend_capabilities`

Return borrowed capability descriptors that describe backend behavior in a
machine-readable way, including determinism hints, training support, fused dense
paths and whether prepared inference is preferred.

### `nm_set_seed`

Sets RNG seed for reproducible initialization/training behavior.
