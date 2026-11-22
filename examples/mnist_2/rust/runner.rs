use std::convert::TryInto;
use std::env;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

fn read_f32_binary(path: &str, expected_count: usize) -> Result<Vec<f32>, std::io::Error> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    if bytes.len() != expected_count * 4 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("Expected {} bytes, got {}", expected_count * 4, bytes.len()),
        ));
    }
    let mut floats = Vec::with_capacity(expected_count);
    for chunk in bytes.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Slice convert failed")
        })?;
        floats.push(f32::from_ne_bytes(arr));
    }
    Ok(floats)
}

fn read_u8_binary(path: &str, expected_count: usize) -> Result<Vec<u8>, std::io::Error> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    if bytes.len() != expected_count {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("Expected {} bytes, got {}", expected_count, bytes.len()),
        ));
    }
    Ok(bytes)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let inputs_path = args.get(1).map(|s| s.as_str()).unwrap_or("mnist_test_inputs.bin");
    let c_preds_path = args.get(2).map(|s| s.as_str()).unwrap_or("mnist_c_predictions.bin");
    let labels_path = args.get(3).map(|s| s.as_str()).unwrap_or("mnist_test_labels.bin");
    let num_samples: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let bench_iters: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(20);

    let input_dim = 784;
    let num_classes = 10;

    let inputs = read_f32_binary(inputs_path, num_samples * input_dim)
        .expect("Failed to read test inputs");
    let c_preds = read_f32_binary(c_preds_path, num_samples * num_classes)
        .expect("Failed to read C predictions");
    let labels = read_u8_binary(labels_path, num_samples)
        .expect("Failed to read labels");

    // 1. Correctness run
    let rust_preds = run_nm_frontend(&inputs);
    assert_eq!(
        rust_preds.len(),
        num_samples * num_classes,
        "Mismatch in output prediction dimensions"
    );

    let mut max_delta: f32 = 0.0;
    let mut correct_rust: usize = 0;
    let mut class_mismatches_vs_c: usize = 0;

    for i in 0..num_samples {
        let label = labels[i] as usize;

        let mut r_best_class = 0;
        let mut r_max_val = f32::NEG_INFINITY;

        let mut c_best_class = 0;
        let mut c_max_val = f32::NEG_INFINITY;

        for c in 0..num_classes {
            let idx = i * num_classes + c;
            let rv = rust_preds[idx];
            let cv = c_preds[idx];
            let delta = (rv - cv).abs();
            if delta > max_delta {
                max_delta = delta;
            }

            if rv > r_max_val {
                r_max_val = rv;
                r_best_class = c;
            }
            if cv > c_max_val {
                c_max_val = cv;
                c_best_class = c;
            }
        }

        if r_best_class == label {
            correct_rust += 1;
        }
        if r_best_class != c_best_class {
            class_mismatches_vs_c += 1;
        }
    }

    let accuracy = (correct_rust as f64) / (num_samples as f64) * 100.0;
    println!("\n=== GraphIR Generated Rust Correctness ===");
    println!("  Samples checked       : {}", num_samples);
    println!("  Max delta vs nm_lib C : {:.2e}", max_delta);
    println!("  Class mismatches vs C : {}", class_mismatches_vs_c);
    println!("  Rust Accuracy         : {:.2}%", accuracy);

    assert!(
        max_delta <= 1e-4,
        "Rust predictions deviated from C reference by {:e} (> 1e-4)",
        max_delta
    );
    assert_eq!(
        class_mismatches_vs_c, 0,
        "Rust predictions have class mismatch vs C reference"
    );

    // 2. Benchmark run
    let mut total_duration = std::time::Duration::ZERO;
    for _ in 0..bench_iters {
        let start = Instant::now();
        let _ = run_nm_frontend(&inputs);
        total_duration += start.elapsed();
    }

    let mean_latency_ms = total_duration.as_secs_f64() * 1000.0 / (bench_iters as f64);
    let sample_latency_ms = mean_latency_ms / (num_samples as f64);
    let throughput = (num_samples as f64 * bench_iters as f64) / total_duration.as_secs_f64();

    println!("\n=== GraphIR Generated Rust Benchmark Results ===");
    println!("  Samples per batch : {}", num_samples);
    println!("  Iterations        : {}", bench_iters);
    println!("  Mean Batch Latency: {:.2} ms", mean_latency_ms);
    println!("  Latency per sample: {:.4} ms", sample_latency_ms);
    println!("  Throughput        : {:.1} samples/sec", throughput);
}
