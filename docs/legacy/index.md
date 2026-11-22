# Legacy `nm.h` API

!!! warning "Legacy API"
    `nm.h` is the older, matrix-based entry point. It keeps working and shares
    the trainer with the current API, but new programs should use
    [`nm_api.h`](../api/overview.md). The handle of `nm.h` is `NmModel`; the
    handle of `nm_api.h` is `NmNetwork`.

## When to use it

- You already have code written against `nm_create` / `nm_compile` / `nm_fit`.
- You need the low-level helpers that only `nm.h` exposes: activation and
  loss kernels on `FloatMatrix`, `nm_set_backend`, `nm_set_tensor_layout`,
  `nm_set_autograd_mode`, `nm_set_batch_parallel`.
- You read or write the `NMS3` binary format (`nm_save`, `nm_load`).

`nm.h` includes `sm.h` from the mMatrix repository and works on `FloatMatrix`
values, which is why it needs the mMatrix include path in every program.

## Workflow

1. Define the input shape with `NmShape`
2. `nm_create(...)`
3. Add layers with `nm_add_dense`, `nm_add_conv2d`, `nm_add_maxpool2d`,
   `nm_add_flatten` (`nm_add_avgpool2d` and `nm_add_batchnorm2d` are declared
   but not implemented)
4. `nm_compile(...)` with an `NmCompileConfig` (loss)
5. `nm_fit(...)` with an `NmTrainConfig`
6. `nm_predict(...)`, `nm_evaluate(...)`, `nm_accuracy(...)`
7. `nm_save(...)` / `nm_load(...)`, then `nm_free(...)`

```c
#include "nm.h"

NmModel *model = nm_create((NmShape){.ndim = 1, .dims = {4}});
nm_add_dense(model, &(NmDenseConfig){.units = 8, .activation = NM_ACT_RELU});
nm_add_dense(model, &(NmDenseConfig){.units = 3, .activation = NM_ACT_SOFTMAX});

NmCompileConfig compile_cfg = nm_compile_defaults();
compile_cfg.loss_type = NM_LOSS_CROSS_ENTROPY;
nm_compile(model, &compile_cfg);

NmTrainConfig train_cfg = nm_train_defaults();
train_cfg.epochs = 200;
train_cfg.batch_size = 16;
train_cfg.learning_rate = 0.01f;
nm_fit(model, X, Y, &train_cfg);          /* X, Y: FloatMatrix * */

float accuracy = 0.0f;
nm_evaluate(model, X, Y, &accuracy);
nm_free(model);
```

## Pages

- [Lifecycle & configuration](lifecycle.md)
- [Training & inference](training.md)
- [Activations & derivatives](activations.md)
- [Losses](losses.md)
- [Metrics & utilities](metrics.md)
- [Serialization (NMS3)](serialization-nms3.md)
- [MNIST with `nm.h`](mnist.md)

Most programs under `examples/` (`xor`, `iris`, `boston`, `fashion`, `cifar`,
`mnist`) still use this API. `examples/mnist_api` is the `nm_api.h` version.
