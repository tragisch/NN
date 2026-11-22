load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

exports_files([
  "data/t10k-images-idx3-ubyte",
  "data/t10k-labels-idx1-ubyte",
  "data/train-images-idx3-ubyte",
  "data/train-labels-idx1-ubyte",
], visibility = ["//visibility:public"])

cc_binary(
    name = "fashion",
    srcs = ["main.c"],
    data = [
      "data/t10k-images-idx3-ubyte",
      "data/t10k-labels-idx1-ubyte",
      "data/train-images-idx3-ubyte",
      "data/train-labels-idx1-ubyte",
    ],
    deps = [
      "//app/nm_lib:nm_api",
      ":fashion_data",
      "//third_party/log.c:log",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "fashion_data",
    srcs = ["fashion_data.c", "fashion_data.h"],
    hdrs = ["fashion_data.h"],
    includes = ["."],
    deps = [
      "//app/nm_lib:nm_api",
      "//third_party/log.c:log",
    ],
    visibility = ["//visibility:public"],
)
