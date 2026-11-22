load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

exports_files([
    "gspc.csv",
], visibility = ["//visibility:public"])

cc_binary(
    name = "sp500",
    srcs = ["main.c"],
    data = [
        "gspc.csv",
    ],
    deps = [
        "//app/nm_lib:nm_api",
        "//third_party/log.c:log",
    ],
    visibility = ["//visibility:public"],
)
