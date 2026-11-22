load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_binary(
    name = "bench_nm_functions",
    srcs = ["bench_nm_functions.c"],
    deps = [
        "//app/nm_lib:nm",
        ":simple_bench", 
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "simple_bench",
    srcs = [
        "simple_bench.h",
    ],
    copts = [
        "-O2",
        "-Wno-macro-redefined"
    ],
)