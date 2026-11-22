#!/usr/bin/env bash
set -euo pipefail

GRAPHIR_ROOT="${GRAPHIR_ROOT:-}"
NN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EPOCHS="${1:-5}"
SAMPLES="${2:-1000}"
BENCH_ITERS="${3:-20}"

echo "================================================================================"
echo " MNIST End-to-End Pipeline: nm_lib (C) -> GraphIR -> Rust & Core AI"
echo " Configuration: epochs=${EPOCHS}, eval_samples=${SAMPLES}, bench_iters=${BENCH_ITERS}"
echo "================================================================================"

cd "$NN_ROOT"

# Step 1: Train & Export via C Engine (nm_lib)
echo -e "\n[1/4] Building & Running nm_lib (C Engine)..."
bazel build //examples/mnist_2:mnist
./bazel-bin/examples/mnist_2/mnist \
    --epochs "$EPOCHS" \
    --eval-samples "$SAMPLES" \
    --bench-iters "$BENCH_ITERS" \
    --export "$NN_ROOT/mnist.graphir.json"

# Step 2: Generate Rust & Core AI code with GraphIR
echo -e "\n[2/4] Generating Rust and Core AI code from GraphIR..."
if [[ ! -d "$GRAPHIR_ROOT" ]]; then
    echo "ERROR: GraphIR root not found at $GRAPHIR_ROOT" >&2
    exit 1
fi

(cd "$GRAPHIR_ROOT" && bazel run //:graphir -- rust "$NN_ROOT/mnist.graphir.json") > "$NN_ROOT/examples/mnist_2/rust/mnist_generated.rs"
(cd "$GRAPHIR_ROOT" && bazel run //:graphir -- coreai "$NN_ROOT/mnist.graphir.json") > "$NN_ROOT/examples/mnist_2/coreai/mnist_coreai.py"

# Step 3: Compile & Benchmark Rust Code
echo -e "\n[3/4] Compiling & Running GraphIR Rust Solution..."
cat "$NN_ROOT/examples/mnist_2/rust/mnist_generated.rs" "$NN_ROOT/examples/mnist_2/rust/runner.rs" > "$NN_ROOT/examples/mnist_2/rust/combined_runner.rs"
rustc --edition 2021 -O -C target-cpu=native "$NN_ROOT/examples/mnist_2/rust/combined_runner.rs" -o "$NN_ROOT/examples/mnist_2/rust/runner_bin"
"$NN_ROOT/examples/mnist_2/rust/runner_bin" \
    "$NN_ROOT/mnist_test_inputs.bin" \
    "$NN_ROOT/mnist_c_predictions.bin" \
    "$NN_ROOT/mnist_test_labels.bin" \
    "$SAMPLES" \
    "$BENCH_ITERS"

# Step 4: Build .aimodel & Benchmark Core AI
echo -e "\n[4/4] Building & Running Apple Core AI Solution..."
uv run --project "$GRAPHIR_ROOT/tools/coreai" --python 3.13 \
    "$NN_ROOT/examples/mnist_2/coreai/mnist_coreai.py" "$NN_ROOT/examples/mnist_2/coreai/mnist.aimodel"

uv run --project "$GRAPHIR_ROOT/tools/coreai" --python 3.13 python \
    "$NN_ROOT/examples/mnist_2/coreai/benchmark_coreai.py" \
    --model "$NN_ROOT/examples/mnist_2/coreai/mnist.aimodel" \
    --inputs "$NN_ROOT/mnist_test_inputs.bin" \
    --c-preds "$NN_ROOT/mnist_c_predictions.bin" \
    --labels "$NN_ROOT/mnist_test_labels.bin" \
    --samples "$SAMPLES" \
    --bench-iters "$BENCH_ITERS"

echo -e "\n================================================================================"
echo " All 3 solutions successfully verified and benchmarked!"
echo "================================================================================"
