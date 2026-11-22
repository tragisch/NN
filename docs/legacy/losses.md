# API: Loss Functions

!!! warning "Legacy API"
    This page describes the legacy `nm.h` API, which stays available but is not
    the recommended entry point. New programs should use `nm_api.h`; see
    [API Overview](../api/overview.md).

Loss functions return scalar objective values from predicted and target matrices.

## `nm_loss_mse`

Purpose:

- Mean Squared Error for regression-like objectives.

Benefit:

- Simple, stable baseline metric.

Notes:

- Good fit for regression.
- Less suitable for multi-class probability training than cross-entropy.

## `nm_loss_cross_entropy`

Purpose:

- Cross-entropy for probability outputs (typically softmax logits after activation).

Benefit:

- Better gradient behavior for multi-class classification.

Notes:

- Usually paired with softmax outputs.
- `nm_compile(...)` enables the fused softmax + cross-entropy shortcut automatically when the output layer uses `NM_ACT_SOFTMAX`.

## Choosing a loss

- Use `NM_LOSS_CROSS_ENTROPY` for classification.
- Use `NM_LOSS_MSE` for regression-oriented tasks.

When using softmax outputs, the fused shortcut is applied automatically during compile.
