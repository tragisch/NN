load("//:tools/matrix_build.bzl", "matrix_cc_binary")

matrix_cc_binary(
    name = "bench_nm_relu",
    srcs = ["benchmark_nm_relu.cpp"],
    deps = [
        "@google_benchmark//:benchmark",
        "//app/nm_lib:nm",
    ],
)

matrix_cc_binary(
    name = "bench_nm_mnist",
    srcs = ["benchmark_nm_mnist.cpp"],
    data = [
        "//examples/mnist:t10k-images-idx3-ubyte",
        "//examples/mnist:t10k-labels-idx1-ubyte",
        "//examples/mnist:train-images-idx3-ubyte",
        "//examples/mnist:train-labels-idx1-ubyte",
    ],
    deps = [
        "@google_benchmark//:benchmark",
        "//examples/mnist:mnist_data",
        "//app/nm_lib:nm",
    ],
)

# Microbenchmark: Canonical mmatrix tensor Conv2D performance cases
matrix_cc_binary(
    name = "check_mmatrix_perf",
    srcs = ["check_mmatrix_perf.cpp"],
    deps = [
        "@google_benchmark//:benchmark",
        "@mmatrix//app/tensor:st",
    ],
)

# Microbenchmark: Fine-grained operation profiling
matrix_cc_binary(
    name = "bench_nm_operations",
    srcs = ["bench_nm_operations.cpp"],
    deps = [
        "@google_benchmark//:benchmark",
        "//app/nm_lib:nm",
    ],
)

matrix_cc_binary(
    name = "bench_nm_seq_conv",
    srcs = ["benchmark_nm_seq_conv.cpp"],
    deps = [
        "@google_benchmark//:benchmark",
        "//app/nm_lib:nm",
    ],
)