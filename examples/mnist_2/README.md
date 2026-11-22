# MNIST 2: Frontend, C-Engine, GraphIR (Rust & Apple Core AI)

Demonstrates end-to-end training and inference export using `nm_lib`'s modular architecture:
1. **`nm_frontend` + `nm_engine_c`**: Model construction, training, inference and benchmarking in C.
2. **`nm_export_graphir`**: Exporting the trained dense topology to portable GraphIR v3 JSON.
3. **GraphIR -> Rust**: Automatic code generation of standalone, dependency-free Rust inference code.
4. **GraphIR -> Apple Core AI (`corai`)**: Automatic code generation of Core AI Python authoring scripts and compilation to `.aimodel`.
5. **Cross-Engine Verification & Benchmarks**: Correctness assertion ($\Delta \le 10^{-4}$, exact class parity) and performance comparison across all 3 execution paths.

## Architecture

```text
nm_frontend (C) ──> nm_engine_c (Fit & Inference)
      │
      └──> nm_export_graphir (JSON v3)
                 │
                 ├──> GraphIR CLI -- rust   ──> Rust Code (rustc -O)
                 │
                 └──> GraphIR CLI -- coreai ──> Apple Core AI (.aimodel)
```

## Quickstart

Run the complete training, code generation, verification, and benchmark pipeline:

```bash
./examples/mnist_2/run_and_benchmark.sh [epochs] [eval_samples] [bench_iters]
```

Example:
```bash
./examples/mnist_2/run_and_benchmark.sh 5 1000 20
```

## Manual Steps

### 1. Build and Run C Engine
```bash
bazel run //examples/mnist_2:mnist -- --epochs 5 --eval-samples 1000 --export mnist.graphir.json
```

### 2. Export & Run Rust Code
```bash
GRAPHIR_ROOT=/path/to/GraphIR
(cd "$GRAPHIR_ROOT" && bazel run //:graphir -- rust "$PWD/mnist.graphir.json") > examples/mnist_2/rust/mnist_generated.rs
cat examples/mnist_2/rust/mnist_generated.rs examples/mnist_2/rust/runner.rs > examples/mnist_2/rust/combined_runner.rs
rustc --edition 2021 -O examples/mnist_2/rust/combined_runner.rs -o examples/mnist_2/rust/runner_bin
./examples/mnist_2/rust/runner_bin mnist_test_inputs.bin mnist_c_predictions.bin mnist_test_labels.bin 1000 20
```

### 3. Export & Run Apple Core AI
```bash
(cd "$GRAPHIR_ROOT" && bazel run //:graphir -- coreai "$PWD/mnist.graphir.json") > examples/mnist_2/coreai/mnist_coreai.py
uv run --project "$GRAPHIR_ROOT/tools/coreai" --python 3.13 examples/mnist_2/coreai/mnist_coreai.py examples/mnist_2/coreai/mnist.aimodel
uv run --project "$GRAPHIR_ROOT/tools/coreai" --python 3.13 python examples/mnist_2/coreai/benchmark_coreai.py \
    --model examples/mnist_2/coreai/mnist.aimodel \
    --inputs mnist_test_inputs.bin \
    --c-preds mnist_c_predictions.bin \
    --labels mnist_test_labels.bin \
    --samples 1000 \
    --bench-iters 20
```
