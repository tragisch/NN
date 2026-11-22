load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

exports_files([
  "data/data_batch_1.bin",
  "data/data_batch_2.bin",
  "data/data_batch_3.bin",
  "data/data_batch_4.bin",
  "data/data_batch_5.bin",
  "data/test_batch.bin",
  "data/batches.meta.txt",
], visibility = ["//visibility:public"])

cc_binary(
    name = "cifar",
    srcs = ["main.c"],
    data = [
      "data/data_batch_1.bin",
      "data/data_batch_2.bin",
      "data/data_batch_3.bin",
      "data/data_batch_4.bin",
      "data/data_batch_5.bin",
      "data/test_batch.bin",
    ],
    # CIFAR-10 raw data (~170 MB) is not in git; download manually.
    # Tagged manual so `bazel build //...` skips it in CI.
    tags = ["manual"],
    deps = [
      "//app/nm_lib:nm_api",
      ":cifar_data",
      "//third_party/log.c:log",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cifar_data",
    srcs = ["cifar_data.c", "cifar_data.h"],
    hdrs = ["cifar_data.h"],
    includes = ["."],
    deps = [
      "//app/nm_lib:nm_api",
      "//third_party/log.c:log",
    ],
    visibility = ["//visibility:public"],
)
