load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

exports_files([
  "t10k-images-idx3-ubyte",
  "t10k-labels-idx1-ubyte",
  "train-images-idx3-ubyte",
  "train-labels-idx1-ubyte",
], visibility = ["//visibility:public"])


cc_binary(
    name = "mnist",
    srcs = ["main.c"],
    data = [
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
    ],
    deps = [
        ":mnist_data",
        "//app/nm_lib:nm_engine_c",
        "//app/nm_lib:nm_export_graphir",
        "//app/nm_lib:nm_frontend",
        "//third_party/log.c:log",
        "@argtable3//:argtable3",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "main_simple",
    srcs = ["main_simple.c"],
    data = [
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
    ],
    deps = [
        ":mnist_data",
        "//app/nm_lib:nm_engine_c",
        "//app/nm_lib:nm_frontend",
        "//third_party/log.c:log",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mnist_data",
    srcs = [
        "mnist_data.c",
        "mnist_data.h",
    ],
    hdrs = ["mnist_data.h"],
    includes = ["."],
    deps = [
        "//app/nm_lib:nm_frontend",
        "//third_party/log.c:log",
    ],
    visibility = ["//visibility:public"],
)
cc_binary(
    name = "bench_cnn",
    srcs = ["bench_cnn.c"],
    data = [
        "//:mnist_test_inputs.bin",
        "//:mnist_test_labels.bin",
    ],
    deps = [
        "//app/nm_lib:nm_engine_c",
        "//app/nm_lib:nm_export_graphir",
        "//app/nm_lib:nm_frontend",
        "//third_party/log.c:log",
        "@mmatrix//app/matrix:sm",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "prof_cnn",
    srcs = ["prof_cnn.c"],
    deps = [
        "@mmatrix//app/tensor:st",
        "//third_party/log.c:log",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "prof_parts",
    srcs = ["prof_parts.c"],
    deps = [
        "@mmatrix//app/matrix:sm",
        "//third_party/log.c:log",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "prof_exact",
    srcs = ["prof_exact.c"],
    deps = [
        "@mmatrix//app/matrix:sm",
        "//third_party/log.c:log",
    ],
    visibility = ["//visibility:public"],
)
