# Library Usage

How the pieces of `nm_api.h` fit together, what the defaults are and where the
sharp edges are.

## Data layout

All data is row-major `float` (float32), one row per example.

```c
typedef struct {
  float *data;
  size_t ndim;       /* 2..4 */
  size_t dims[4];    /* dims[0] = number of examples */
} NmTensor;
```

- `nm_tensor_2d(data, rows, cols)` wraps a matrix: `rows` examples with `cols`
  features each. Your array must hold `rows * cols` floats.
- Images use `ndim = 4`, `{n, channels, height, width}` (NCHW). For a dense
  first layer a flat `{n, features}` tensor is equally valid, and for a
  convolutional first layer `{n, c * h * w}` works as well because the network
  reads each example as one row.
- Tensors you pass in are **borrowed**: never copied, never freed by the
  library. Tensors the library returns (from `predict`) are **yours**; free
  them with `nm_tensor_free`.
- Targets have the same number of rows as the inputs. For classification they
  are one-hot rows (`n x classes`); for regression the values themselves.

There is no loader in `nm_api.h`. The shipped examples read their datasets
with helpers in `examples/*/*_data.c`; for a CSV file, parse into a
`float` array yourself and normalize each feature (zero mean, unit variance
trains much better than raw values).

## Describing a network

```c
NmSpec *spec = nm_spec_new((NmShape){.ndim = 1, .dims = {784}});
nm_spec_dense(spec, 128, NM_ACT_RELU);
nm_spec_dense(spec, 10, NM_ACT_SOFTMAX);
```

- Input shape: one example, 1 to 3 dimensions: `{features}` or
  `{channels, height, width}`.
- Layers: `nm_spec_dense`, `nm_spec_conv2d(filters, kernel, stride, padding,
  activation)`, `nm_spec_maxpool2d(kernel, stride)`, `nm_spec_flatten`.
- Activations: `NM_ACT_NONE` (linear), `RELU`, `SIGMOID`, `TANH`, `SOFTMAX`.
- Shapes are checked at build time, not at `nm_spec_*` time. A rejected layer
  shows up as `nm_network_build: layer N rejected ...`.

## Building

```c
NmBuildOptions build = nm_build_options_default();
build.seed = 42;
NmNetwork *net = nm_network_build(spec, &build);   /* NULL on error */
```

`seed = 0` keeps the library's current random state. A non-zero seed also
reseeds the library-wide generator used for weight initialization, so build
networks one after another when you need reproducible runs.

## Training

```c
NmFitOptions fit = nm_fit_options_default();
fit.epochs = 20;
nm_network_fit(net, inputs, &targets, &fit, progress_callback, user_data);
```

| Field | Default | Meaning |
| --- | --- | --- |
| `loss` | `NM_LOSS_NONE` | none selects cross-entropy when targets are given, MSE for an autoencoder |
| `optimizer` | `NM_OPT_SGD` | `SGD`, `MOMENTUM`, `RMSPROP`, `ADAM` |
| `learning_rate` | `0.1` | Adam usually wants `0.001` to `0.01` |
| `batch_size` | `32` | rows per update |
| `epochs` | `10` | passes over the training rows |
| `shuffle` | `true` | reshuffles every epoch; works on a copy, your data stays in order |
| `validation_split` | `0` | fraction of the **last** rows held out; they are not shuffled in |

Things to know:

- **Say the loss for regression.** With `NM_LOSS_NONE` and targets the library
  trains and reports cross-entropy, which is meaningless on continuous values
  (the reported loss can go negative). Set `fit.loss = NM_LOSS_MSE`.
- **Order your data before `validation_split`.** The held-out rows are the
  last ones. If your file is sorted by class or by time, shuffle it first.
- **The progress callback costs a forward pass.** It receives the loss on the
  training rows and, with `validation_split`, on the held-out rows
  (`NAN` without). Return non-zero to stop early; `fit` then returns
  `NM_STATUS_OK`. The callback runs on the calling thread.
- **Autoencoders:** pass `targets = NULL`; the target is the input.
- Calling `fit` again continues from the current weights. Changing `loss` or
  `optimizer` between calls restarts the optimizer state.

## Predicting and evaluating

```c
NmTensor out;
if (nm_network_predict(net, rows, &out) == NM_STATUS_OK) {
  /* out.dims = {2, {n, outputs}}, out.data holds n * outputs floats */
  nm_tensor_free(&out);
}

NmMetrics m;
nm_network_evaluate(net, rows, targets, NM_LOSS_CROSS_ENTROPY, &m);
/* m.loss; m.accuracy is -1 unless the loss is cross-entropy */
```

`predict` allocates the result on every call. Predict a batch of rows at a
time rather than row by row.

Models with several inputs, or with integer or boolean inputs (token ids and
a padding mask, say), use `nm_network_predict_inputs` with `NmInput` values;
`nm_network_input_count` and `nm_network_input_info` describe what a loaded
model expects. `nm_network_predict_batches` runs more examples than the
model's declared batch by chunking. See [Components](../api/components.md).

## Saving and loading

```c
nm_network_save(net, "model_dir");          /* model.graphir.json + weights.girw */
NmNetwork *again = nm_network_load("model_dir");
NmNetwork *foreign = nm_network_import_graphir("path/to/model.graphir.json");
nm_network_export_graphir(net, "single_file.json");   /* weights inline */
```

`save` creates only the last directory level; parent directories must exist.
Loaded networks predict, and chains of the supported layers can also be
trained further. Details in [Serialization](../api/serialization.md).

## Errors

Every call returns an `NmStatus` (or `NULL` for handles and loaders). The
readable message for the calling thread is `nm_last_error()`:

```c
if (!net) {
  fprintf(stderr, "build failed: %s\n", nm_last_error());
}
```

The message stays until the next failing call on the same thread.
Statuses: `INVALID_ARGUMENT`, `IO_ERROR`, `ALLOC_FAILED`, `FORMAT_ERROR`,
`UNSUPPORTED_TYPE`, `INTERNAL_ERROR`. The library never aborts the program on
bad input. Most checks name the offending argument and sizes.

## Threads

- One `NmNetwork` can be shared between threads; calls take turns on an
  internal lock, so predictions on one network do not run in parallel. For
  parallel prediction build one network per thread (same spec, or load the
  saved model).
- Different networks are independent.
- `nm_last_error()` is per thread.
- The progress callback runs in the training thread. Switch to your UI thread
  yourself.

## Reproducibility

Same seed, same data, `deterministic = true` (the default) give the same
weights. Build and train sequentially if you need that guarantee for several
networks, because the initial weights come from a library-wide generator.
