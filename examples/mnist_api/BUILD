load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

cc_binary(
    name = "mnist_api",
    srcs = ["main.c"],
    data = [
        "//examples/mnist:t10k-images-idx3-ubyte",
        "//examples/mnist:t10k-labels-idx1-ubyte",
        "//examples/mnist:train-images-idx3-ubyte",
        "//examples/mnist:train-labels-idx1-ubyte",
    ],
    deps = [
        "//app/nm_lib:nm_api",
        "//examples/mnist:mnist_data",
    ],
)
