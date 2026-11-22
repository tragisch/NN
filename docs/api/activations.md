# API: Activations

Activations are chosen per layer in the spec (`nm_spec_dense`, `nm_spec_conv2d`).

| `NmActivation` | Formula | Typical use |
| --- | --- | --- |
| `NM_ACT_NONE` | identity | regression output, logits |
| `NM_ACT_RELU` | `max(0, x)` | hidden layers |
| `NM_ACT_SIGMOID` | `1 / (1 + e^-x)` | bounded `[0, 1]` outputs, binary probabilities |
| `NM_ACT_TANH` | `tanh(x)` | smooth regression, bounded `[-1, 1]` |
| `NM_ACT_SOFTMAX` | row-wise, numerically stabilized | multi-class output with cross-entropy |

Hints:

- Use a linear output (`NM_ACT_NONE`) with `NM_LOSS_MSE` for regression.
- Softmax is meant for the last layer; with cross-entropy the combined
  gradient is used.
- Smooth targets (curves, physical models) often train better with `TANH`
  than with `RELU` on small networks.
- Not available: LeakyReLU, ELU, GELU (GELU exists only inside imported
  GraphIR graphs, inference only), Swish.

The element-wise kernels and their derivatives on `FloatMatrix` (`nm_relu`,
`nm_relu_grad`, ...) are part of the [legacy API](../legacy/activations.md).
