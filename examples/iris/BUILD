load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

exports_files([
  "data/iris.data",
], visibility = ["//visibility:public"])

cc_binary(
    name = "iris",
    srcs = ["main.c"],
    data = [
        "data/iris.data",
    ],
    deps = [
        "//app/nm_lib:nm_api",
        "//third_party/log.c:log",
        ":iris_data",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "iris_data",
    srcs = ["iris_data.c"],
    hdrs = ["iris_data.h"],
    includes = ["."],
    deps = [
        "//app/nm_lib:nm_api",
        "//third_party/log.c:log",
    ],
    visibility = ["//visibility:public"],
)
