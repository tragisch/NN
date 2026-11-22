# API: Saving, Loading and GraphIR

nm_lib persists networks as **GraphIR**, a portable graph format shared with
the GraphIR project (checker, Rust code generator). There is no separate
binary format: a saved model is a JSON graph plus, optionally, a checksummed
weight package.

## Save and load

```c
NmStatus   nm_network_save(const NmNetwork *model, const char *dir);
NmNetwork *nm_network_load(const char *dir);
```

`nm_network_save` creates `dir` (one level; the parent must exist) and writes

| File | Content |
| --- | --- |
| `model.graphir.json` | GraphIR v4 graph: topology, shapes, operations |
| `weights.girw` | weights as a `.girw` version 2 package, XXH64 checksums, 64-byte aligned blobs |

The JSON refers to the package by file name, so keep both files in one
directory. `nm_network_load(dir)` reads them back and verifies the checksums.
Errors: `NM_STATUS_IO_ERROR` for unreadable files, `NM_STATUS_FORMAT_ERROR` for
malformed or corrupt content. The message is in `nm_last_error()`.

The graph is exported with a declared batch of one. A loaded model with a
single input accepts any number of rows in `predict`.

## One self-contained file

```c
NmStatus nm_network_export_graphir(const NmNetwork *model, const char *path);
```

Writes a single JSON file with the weights inline (float32 only). Convenient
for small models and for handing a model to the GraphIR tools.

## Import models from elsewhere

```c
NmNetwork *nm_network_import_graphir(const char *path);
```

Loads any GraphIR v3 or v4 document, with inline weights or with an external
`.girw` package that is resolved relative to the document. The model must use
operations from the capability profile ([Components](components.md#capabilities)).
Graphs that are layer chains (`Dense`, `Conv2D`, `MaxPool2D`, `Flatten`) can
be trained further. Graphs with residual connections, attention, `LayerNorm`,
`Gather` and similar operations run through an interpreter for prediction and
cannot be trained.

Float16 and bfloat16 graphs are converted exactly to float32 when loaded and
computed in float32 (promotion). Training them is rejected with
`NM_STATUS_UNSUPPORTED_TYPE`. Quantization is not supported.

## Check with GraphIR tools

From a GraphIR checkout:

```text
bazel run //:graphir -- check /absolute/path/to/model.graphir.json
bazel run //:graphir -- rust  /absolute/path/to/model.graphir.json > model.rs
```

See `examples/graphir_export/README.md` and `tools/verify_graphir_export.sh`
for the full cross-repository check against generated Rust.

## Lower-level headers

| Header | Provides |
| --- | --- |
| `nm_export_graphir.h` | `nm_export_graphir_json` (frontend model), `nm_export_graphir_graph_json` (any graph), `NmGraphIrExportOptions{batch_size, weights_path}`; `weights_path == NULL` writes inline |
| `nm_import_graphir.h` | `nm_import_graphir_json`, `nm_graph_model_free` |
| `nm_girw.h` | open, verify and write `.girw` packages, from files or memory |

Opsets: Dense models export as opset 1; convolution, pooling and batched
matrix products need opset 2. Import accepts opsets 1 to 3 and format
versions 3 and 4.

## Older formats

`.nms*` files written by the legacy API can be read with
`nm_frontend_import_legacy` (topology and weights only), and the `NMS3`
format of `nm_save` is described in the
[legacy serialization page](../legacy/serialization-nms3.md).
