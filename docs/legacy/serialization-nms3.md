# API: Serialization

!!! warning "Legacy API"
    This page describes the legacy `nm.h` API, which stays available but is not
    the recommended entry point. New programs should use `nm_api.h`; see
    [API Overview](../api/overview.md).

Models are persisted through the unified save/load entry points:

```c
NmStatus nm_save(const NmModel *model, const char *path);
NmStatus nm_save_inference(const NmModel *model, const char *path);
NmStatus nm_load(const char *path, NmModel **out_model);
```

Internally, the current format is the **NMS3** chunked binary format.
`nm_load(...)` still accepts legacy `NMS2` files for backwards compatibility.

Both save functions return `NM_STATUS_OK` on success, `NM_STATUS_IO_ERROR` for file errors, and `NM_STATUS_FORMAT_ERROR` when the file is invalid. The loaded model is heap-allocated; release it with `nm_free`.

---

## NMS3 file layout

```
┌────────────────────────────────────┐
│  File header   (8 bytes)           │
├────────────────────────────────────┤
│  Chunk META    (aligned payload)   │
├────────────────────────────────────┤
│  Chunk CONF    (aligned payload)   │
│  Chunk LAYS    (aligned payload)   │
│  Chunk TENS    (aligned payload)   │
│  Chunk WGHZ    (aligned payload)   │
│  Chunk PLAN    (aligned payload)   │
│  Chunk TRNG    (aligned payload,
│                 optional)          │
│  Chunk HASH    (aligned payload)   │
└────────────────────────────────────┘
```

All chunk payloads are padded to the next multiple of the format alignment
(currently `32` bytes). Unknown chunks are skipped, so the format can grow
without breaking older readers.

---

## File header

| Offset | Size | Type    | Value  | Description   |
|--------|------|---------|--------|---------------|
| 0      | 4    | char[4] | `NMS3` | Magic bytes   |
| 4      | 4    | u32     | `3`    | Format version|

---

## Chunk format

Every chunk starts with a 12-byte header followed by exactly `payload_length`
bytes of data. Payload length already includes alignment padding. Unknown chunk
IDs are **silently skipped** for forward compatibility.

| Offset | Size | Type    | Description              |
|--------|------|---------|--------------------------|
| 0      | 4    | char[4] | Chunk ID (e.g. `"CONF"`) |
| 4      | 8    | u64     | Payload length in bytes  |

---

## META chunk

The META chunk records explicit format metadata and global flags such as
alignment and whether the file is inference-only.

It also stores a backend name and architecture string (`nm.seq`) for tooling,
inspection and future export pipelines.

## CONF chunk

Stores model-level configuration for shape, compile mode and default runtime
knobs. Inference-only exports keep this chunk but zero out training-specific
settings and omit the optional `TRNG` chunk.

| Byte offset | Size | Type   | Field                                   | Notes                                      |
|-------------|------|--------|-----------------------------------------|--------------------------------------------|
| 0           | 32   | u8[32] | `input_shape`                           | Padded `NmShape` (4 × u64)                 |
| 32          | 4    | u32    | `loss_type`                             | `NmLossType` enum                          |
| 36          | 1    | u8     | derived softmax+cross-entropy shortcut  | internal compile optimization flag         |
| 37          | 4    | u32    | `optimizer_type`                        | `NmOptimizerType` enum                     |
| 41          | 4    | f32    | `learning_rate`                         |                                            |
| 45          | 4    | f32    | `momentum`                              |                                            |
| 49          | 4    | f32    | `beta1`                                 | Adam β₁                                    |
| 53          | 4    | f32    | `beta2`                                 | Adam β₂                                    |
| 57          | 4    | f32    | `epsilon`                               | Adam ε                                     |
| 61          | 4    | u32    | `tensor_layout`                         | `NmTensorLayout`: NCHW=0, NHWC=1           |
| 65          | 8    | u64    | `num_layers`                            |                                            |
| 73          | 4    | u32    | `autograd_mode`                         | `NmAutogradMode`: DISABLED=0, TAPE=1       |
| 77          | 1    | u8     | `batch_parallel.enabled`                | `0` or `1`                                 |
| 78          | 1    | u8     | `batch_parallel.deterministic`          | `0` or `1`                                 |
| 79          | 8    | u64    | `batch_parallel.num_workers`            | `0` = auto-tune at `nm_fit` time           |

**Backward compatibility:** The runtime-config fields at offsets 73–86 were added in format version 2 (current). Loaders detect their presence by comparing remaining bytes in the CONF chunk. Files without these fields load correctly and default to `NM_AUTOGRAD_DISABLED` and batch-parallel disabled.

---

## LAYS chunk

Contains one descriptor per layer in order, now including a standardized,
machine-readable layer name such as `blk.0.dense` or `blk.1.conv2d`.

## TENS chunk

Contains explicit tensor descriptors with standardized names such as
`blk.0.dense.weight`, `blk.0.dense.bias`, `blk.1.conv2d.weight`, etc.

This mirrors the intent behind GGUF's standardized tensor naming: tooling can
reason about model internals without guessing from layer order alone.

### Common layer descriptor header (72 bytes)

| Offset | Size   | Type    | Field          |
|--------|--------|---------|----------------|
| 0      | 4      | u32     | `layer_type`   |
| 4      | 4      | u32     | `activation`   |
| 8      | 32     | u8[32]  | `input_shape`  |
| 40     | 32     | u8[32]  | `output_shape` |

### Type-specific tail

| Layer type          | Extra bytes | Fields (u64 unless noted)                                               |
|---------------------|-------------|-------------------------------------------------------------------------|
| `NM_LAYER_DENSE`    | 8           | `num_outputs`                                                           |
| `NM_LAYER_CONV2D`   | 40          | `out_channels`, `kernel_h`, `kernel_w`, `stride`, `algo(u32)`, `layout(u32)` |
| `NM_LAYER_MAXPOOL2D`| 48          | `pool_h`, `pool_w`, `stride_h`, `stride_w`, `pad_h`, `pad_w`           |
| `NM_LAYER_FLATTEN`  | 0           | *(no tail)*                                                             |

---

## WGHZ chunk

Contains zstd-compressed trainable parameter payloads in layer order.

The NMS3 metadata chunks make the file more explicit even though the actual
weight payload remains compact and simple to parse in C.

### Dense layer

| Size          | Field                      |
|---------------|----------------------------|
| 4             | `layer_type` marker (u32)  |
| 8             | `rows` (u64)               |
| 8             | `cols` (u64)               |
| `rows×cols×4` | weight matrix (f32[])      |
| 8             | `bias_cols` (u64)          |
| `bias_cols×4` | bias vector (f32[])        |

### Conv2D layer

| Size          | Field                      |
|---------------|----------------------------|
| 4             | `layer_type` marker (u32)  |
| 8             | `ndim` (u64)               |
| 4×8           | `shape[0..3]` (u64[4])     |
| `numel×4`     | weight tensor (f32[])      |
| 8             | `bias_numel` (u64)         |
| `bias_numel×4`| bias tensor (f32[])        |

---

## PLAN chunk

Stores the last prepared inference-plan summary seen by the runtime:

- prepared batch rows
- output width
- peak pooled tensor count
- peak pooled tensor bytes

This is diagnostic/runtime metadata, not a hard execution contract.

## TRNG chunk (optional)

Stores training-only runtime state such as optimizer hyperparameters,
optimizer step, autograd mode and batch-parallel settings.

Inference-only exports omit this chunk.

## Legacy support

`nm_load(...)` still accepts legacy `NMS2` files and maps them into the newer
internal state layout.

---

## Example usage

```c
/* Save after training */
NmStatus s = nm_save(model, "model.nms3");
assert(s == NM_STATUS_OK);

/* Load for inference */
NmModel *loaded = NULL;
s = nm_load("model.nms3", &loaded);
assert(s == NM_STATUS_OK);

FloatMatrix *pred = NULL;
nm_predict(loaded, input, &pred);
sm_destroy(pred);
nm_free(loaded);
```

Runtime training config (autograd mode, batch-parallel settings) is restored on
full model saves:

```c
nm_set_autograd_mode(model, NM_AUTOGRAD_TAPE);
nm_set_batch_parallel(model, &(NmBatchParallelConfig){
    .enabled = true, .deterministic = false, .num_workers = 4
});
nm_save(model, "model.nms3");

NmModel *m2 = NULL;
nm_load("model.nms3", &m2);
assert(nm_autograd_mode(m2) == NM_AUTOGRAD_TAPE);
assert(nm_batch_parallel(m2).num_workers == 4);
nm_free(m2);

/* Inference-only export */
nm_save_inference(model, "model-infer.nms3");
nm_load("model-infer.nms3", &m2);
assert(nm_model_is_inference_only(m2));
nm_free(m2);
```
