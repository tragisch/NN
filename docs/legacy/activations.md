# API: Activations & Derivatives

!!! warning "Legacy API"
    This page describes the legacy `nm.h` API, which stays available but is not
    the recommended entry point. New programs should use `nm_api.h`; see
    [API Overview](../api/overview.md).

All activation functions operate on `FloatMatrix` values. Forward activations are in-place.

## Example: forward activation usage

```c
FloatMatrix* logits = sm_create(2, 10);
// ... fill logits

// For hidden layers
nm_relu(logits);

// For output layer probabilities
nm_softmax(logits);
```

## Example: derivative usage during backprop

```c
FloatMatrix* activation = sm_create(32, 128);   // already contains forward output
FloatMatrix* grad_out   = sm_create(32, 128);   // dL/dA from next step

// In-place: grad_out becomes dL/dZ
nm_relu_grad(activation, grad_out);
```

## Forward activations

- `nm_relu(FloatMatrix *mat)`
	- Applies ReLU elementwise: `max(0, x)`.
	- Typical for hidden layers.
- `nm_sigmoid(FloatMatrix *mat)`
	- Applies sigmoid elementwise: $\sigma(x)=\frac{1}{1+e^{-x}}$.
	- Useful for bounded scalar outputs.
- `nm_tanh(FloatMatrix *mat)`
	- Applies tanh elementwise in range `[-1, 1]`.
- `nm_softmax(FloatMatrix *mat)`
	- Applies row-wise softmax to produce class probabilities.
	- Numerically stabilized internally.

Benefits:

- Standard nonlinearities for hidden/output behavior.
- Softmax is numerically stabilized row-wise.

## Derivatives

- `nm_relu_grad(const FloatMatrix *activation, FloatMatrix *grad_output)`
	- In-place gradient mask using forward activation.
- `nm_sigmoid_grad(const FloatMatrix *activation, FloatMatrix *grad_output)`
	- Converts `dL/dA` to `dL/dZ` for sigmoid units.
- `nm_tanh_grad(const FloatMatrix *activation, FloatMatrix *grad_output)`
	- Converts `dL/dA` to `dL/dZ` for tanh units.
- `nm_softmax_ce_grad(const FloatMatrix *predicted, const FloatMatrix *target, FloatMatrix *grad_output)`
	- Efficient combined gradient for softmax + cross-entropy.
	- Avoids explicit Jacobian construction.

Benefits:

- Enables efficient backprop in-place on gradient buffers.
- Softmax+cross-entropy derivative avoids redundant Jacobian computations.

## Utility math

- `nm_tanhf`
- `nm_softmax_denom`

Used internally and useful for validation/testing scenarios.

```c
float v[3] = {1.0f, 2.0f, 3.0f};
float denom = nm_softmax_denom(v, 3);
```
