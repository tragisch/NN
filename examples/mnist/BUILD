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
      "//app/nm_lib:nm_api",
      ":mnist_data",
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
      "//app/nm_lib:nm_api",
      ":mnist_data",
    ],
  visibility = ["//visibility:public"],
)

cc_library(
    name = "mnist_data",
  srcs = ["mnist_data.c", "mnist_data.h"],
  hdrs = ["mnist_data.h"],
    includes = ["."],
    deps = [
      "//app/nm_lib:nm_api",
    ],
  visibility = ["//visibility:public"],
)