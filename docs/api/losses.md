# API: Losses

`NmLoss` selects the training objective and the evaluation metric.

| Value | Use for | Pair with output activation |
| --- | --- | --- |
| `NM_LOSS_MSE` | regression, autoencoders | `NM_ACT_NONE`, or `SIGMOID`/`TANH` for bounded targets |
| `NM_LOSS_CROSS_ENTROPY` | classification with one-hot targets | `NM_ACT_SOFTMAX` |
| `NM_LOSS_NONE` | only in `NmFitOptions`: cross-entropy with targets, MSE without | |

- Softmax with cross-entropy is trained through the fused gradient
  automatically; nothing to switch on.
- `NM_LOSS_NONE` is rejected by `nm_network_evaluate`; name the loss there.
- Cross-entropy expects probabilities in `[0, 1]` and one-hot targets. On
  continuous targets it gives meaningless (even negative) values, so use MSE
  for regression.
- Custom losses are not available through `nm_api.h`
  ([Optimization](optimization.md)).

The scalar helpers `nm_loss_mse` and `nm_loss_cross_entropy` on `FloatMatrix`
belong to the [legacy API](../legacy/losses.md).
