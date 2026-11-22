# API: Metrics

`nm_network_evaluate` returns an `NmMetrics`:

| Field | Meaning |
| --- | --- |
| `loss` | the loss you pass in (`MSE` or `CROSS_ENTROPY`) over all rows |
| `accuracy` | classification only: fraction of rows where the largest output matches the largest target value; `-1` for MSE |

Everything else you compute from `nm_network_predict`. Typical post-processing
for a classifier:

```c
NmTensor out;
nm_network_predict(net, rows, &out);
for (size_t i = 0; i < out.dims[0]; ++i) {
  const float *p = out.data + i * out.dims[1];
  size_t best = 0;
  for (size_t k = 1; k < out.dims[1]; ++k)
    if (p[k] > p[best]) best = k;
  /* best is the predicted class of row i */
}
nm_tensor_free(&out);
```

Not provided: MAE, R², confusion matrix, top-k accuracy. They are a few lines
over the predicted tensor and your targets.

Utilities in the same header: `nm_network_parameter_count`,
`nm_network_summary` ([Describing and Building](lifecycle.md#inspecting)).
