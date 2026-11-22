# API: Training & Inference

## Fit

```c
NmStatus nm_network_fit(NmNetwork *model, NmTensor inputs,
                        const NmTensor *targets, const NmFitOptions *options,
                        NmProgressFn progress, void *user);
```

- `targets == NULL` trains an autoencoder (the target is the input).
- `options == NULL` uses `nm_fit_options_default()`.
- `progress` may be `NULL`.

```c
typedef struct {
  NmLoss loss;
  NmOptimizerType optimizer;
  float learning_rate;
  size_t batch_size;
  size_t epochs;
  bool shuffle;
  float validation_split;
} NmFitOptions;
```

Defaults: `NM_LOSS_NONE`, `NM_OPT_SGD`, `0.1`, `32`, `10`, `true`, `0`. See the
table in [Library Usage](../getting-started/library-usage.md#training).

### Loss selection

| `loss` | targets | used loss |
| --- | --- | --- |
| `NM_LOSS_MSE` | any | mean squared error |
| `NM_LOSS_CROSS_ENTROPY` | any | cross-entropy |
| `NM_LOSS_NONE` | given | cross-entropy |
| `NM_LOSS_NONE` | `NULL` | mean squared error |

Choose explicitly for regression: `NM_LOSS_MSE`.

### Validation checks

`fit` returns `NM_STATUS_INVALID_ARGUMENT` with a message for: a missing
model, `epochs`, `batch_size` or `learning_rate` not positive,
`validation_split` outside `[0, 1)`, input and target row counts that differ,
no training rows left after the split, and weights that are bound read-only
by a connector.

### Progress callback

```c
typedef struct {
  size_t epoch;    /* 1-based */
  size_t epochs;
  float loss;      /* on the training rows after this epoch */
  float val_loss;  /* on the held-out rows, NAN without validation_split */
} NmProgress;

typedef int (*NmProgressFn)(const NmProgress *progress, void *user);
```

Called after every epoch on the calling thread, while the network's lock is
held: do not call other functions on the same network from inside it. A
non-zero return stops
training and `fit` returns `NM_STATUS_OK`. Computing the reported losses costs
one forward pass over the training rows (and the held-out rows) per epoch, so
pass `NULL` when you do not need them.

### What changes the engine

Changing `loss` or `optimizer` between two `fit` calls creates a fresh
engine for the network: the weights stay, the optimizer state restarts.
Calling `fit` repeatedly with the same settings continues training.

## Predict

```c
NmStatus nm_network_predict(const NmNetwork *model, NmTensor inputs,
                            NmTensor *out);
```

Allocates `*out` with shape `{2, {rows, outputs}}`; free it with
`nm_tensor_free`. Failure leaves `*out` zeroed. The error message names the
expected example size, or points to `nm_network_predict_inputs` when the model
has several inputs.

### Several or typed inputs

```c
typedef struct {
  const char *name;      /* GraphIR input name; NULL: match by position */
  NmDType dtype;         /* NM_DTYPE_FLOAT32, NM_DTYPE_INT64, NM_DTYPE_BOOL */
  const void *data;
  size_t ndim;
  size_t dims[NM_INPUT_MAX_DIMS];
} NmInput;

NmStatus nm_network_predict_inputs(const NmNetwork *model, const NmInput *inputs,
                                   size_t count, NmTensor *out);
NmStatus nm_network_predict_batches(const NmNetwork *model, const NmInput *inputs,
                                    size_t count, NmTensor *out);
size_t   nm_network_input_count(const NmNetwork *model);
NmStatus nm_network_input_info(const NmNetwork *model, size_t index, NmInput *out);
```

- `nm_network_input_info` fills name, dtype and declared shape of one input
  (`dims[0] == 0`: any batch size); `out->data` is `NULL`.
- `predict_inputs` needs the declared shapes (a model with a single input also
  accepts any batch size). The output is converted to `float`.
- `predict_batches` takes any number of examples for models whose declared
  shapes fix the batch: it runs chunks of the declared batch and pads the last
  one. Use it only when the examples do not interact, which holds for
  per-sequence encoders such as a transformer with a padding mask. The library
  cannot check that for you.

## Evaluate

```c
typedef struct { float loss; float accuracy; } NmMetrics;

NmStatus nm_network_evaluate(const NmNetwork *model, NmTensor inputs,
                             NmTensor targets, NmLoss loss, NmMetrics *out);
```

`loss` must be `NM_LOSS_MSE` or `NM_LOSS_CROSS_ENTROPY` (no automatic choice
here). `accuracy` is the fraction of rows whose arg-max prediction equals the
arg-max of the target for cross-entropy, and `-1` for MSE.

## Threading

Calls on one network take turns on an internal lock; parallel prediction needs
one network per thread. See [Library Usage](../getting-started/library-usage.md#threads).
