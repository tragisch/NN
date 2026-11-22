load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

exports_files([
    "data/moons.csv",
], visibility = ["//visibility:public"])

cc_binary(
    name = "moons",
    srcs = ["main.c"],
    data = [
        "data/moons.csv",
    ],
    deps = [
        "//app/nm_lib:nm_api",
        "//third_party/log.c:log",
        ":moons_data",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "moons_data",
    srcs = ["moons_data.c"],
    hdrs = ["moons_data.h"],
    includes = ["."],
    deps = [
        "//app/nm_lib:nm_api",
        "//third_party/log.c:log",
    ],
    visibility = ["//visibility:public"],
)
