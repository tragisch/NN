import argparse
import asyncio
from pathlib import Path
import time
import numpy as np
from coreai.runtime import AIModel, NDArray

async def evaluate_and_benchmark(
    model_path: Path,
    inputs_path: Path,
    c_preds_path: Path,
    labels_path: Path,
    samples: int,
    bench_iters: int,
) -> None:
    # 1. Load data
    inputs = np.fromfile(inputs_path, dtype=np.float32).reshape(samples, 784)
    c_preds = np.fromfile(c_preds_path, dtype=np.float32).reshape(samples, 10)
    labels = np.fromfile(labels_path, dtype=np.uint8)

    # 2. Load model
    model = await AIModel.load(model_path)
    function = model.load_function("nm_frontend")

    # 3. Correctness check
    input_tensor = NDArray(inputs)
    outputs = await function({"input": input_tensor})
    coreai_preds = list(outputs.values())[0].numpy()

    max_delta = float(np.max(np.abs(coreai_preds - c_preds)))
    coreai_classes = np.argmax(coreai_preds, axis=1)
    c_classes = np.argmax(c_preds, axis=1)
    mismatches = int(np.sum(coreai_classes != c_classes))
    correct = int(np.sum(coreai_classes == labels))
    accuracy = float(correct) / float(samples) * 100.0

    print("\n=== GraphIR Apple Core AI (corai) Correctness ===")
    print(f"  Samples checked       : {samples}")
    print(f"  Max delta vs nm_lib C : {max_delta:.2e}")
    print(f"  Class mismatches vs C : {mismatches}")
    print(f"  Core AI Accuracy      : {accuracy:.2f}%")

    assert max_delta <= 1e-4, f"Core AI predictions deviated from C reference by {max_delta:e} (> 1e-4)"
    assert mismatches == 0, "Core AI predictions have class mismatch vs C reference"

    # 4. Benchmark runs
    # Warmup
    for _ in range(3):
        _ = await function({"input": input_tensor})

    start_time = time.perf_counter()
    for _ in range(bench_iters):
        _ = await function({"input": input_tensor})
    total_time = time.perf_counter() - start_time

    mean_latency_ms = (total_time / bench_iters) * 1000.0
    sample_latency_ms = mean_latency_ms / samples
    throughput = (samples * bench_iters) / total_time

    print("\n=== GraphIR Apple Core AI (corai) Benchmark Results ===")
    print(f"  Samples per batch : {samples}")
    print(f"  Iterations        : {bench_iters}")
    print(f"  Mean Batch Latency: {mean_latency_ms:.2f} ms")
    print(f"  Latency per sample: {sample_latency_ms:.4f} ms")
    print(f"  Throughput        : {throughput:.1f} samples/sec")

def main():
    parser = argparse.ArgumentParser(description="Benchmark GraphIR Core AI model")
    parser.add_argument("--model", type=Path, default=Path("mnist.aimodel"))
    parser.add_argument("--inputs", type=Path, default=Path("mnist_test_inputs.bin"))
    parser.add_argument("--c-preds", type=Path, default=Path("mnist_c_predictions.bin"))
    parser.add_argument("--labels", type=Path, default=Path("mnist_test_labels.bin"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--bench-iters", type=int, default=20)
    args = parser.parse_args()

    asyncio.run(evaluate_and_benchmark(
        args.model,
        args.inputs,
        args.c_preds,
        args.labels,
        args.samples,
        args.bench_iters,
    ))

if __name__ == "__main__":
    main()
