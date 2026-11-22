# GraphIR export example

This example trains a small XOR classifier through the explicit C-engine API,
prints its predictions and writes `xor.graphir.json` in the current directory.

Run it from the repository root:

```text
bazel run //examples/graphir_export:graphir_export
```

The generated artifact is a static, FP32 GraphIR v3 model with a batch size of
four. It can be checked and turned into dependency-free Rust with GraphIR:

```text
GRAPHIR_ROOT=/path/to/GraphIR tools/verify_graphir_export.sh
```

`verify_graphir_export.sh` independently exports its deterministic fixture;
use the GraphIR CLI directly when validating this example's `xor.graphir.json`:

```text
cd "$GRAPHIR_ROOT"
bazel run //:graphir -- check /absolute/path/to/xor.graphir.json
bazel run //:graphir -- rust /absolute/path/to/xor.graphir.json > xor.rs
```
