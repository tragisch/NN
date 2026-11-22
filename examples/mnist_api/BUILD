load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

cc_binary(
    name = "mnist_api",
    srcs = ["main.c"],
    data = ["//examples/mnist:mnist_data_files"],
    deps = [
        "//app/nm_lib:nm_api",
        "//examples/mnist:mnist_data",
    ],
)
