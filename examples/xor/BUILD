load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

cc_binary(
    name = "xor",
    srcs = ["main.c"],
    deps = [
        "//app/nm_lib:nm_api",
        "//third_party/log.c:log",
    ],
    visibility = ["//visibility:public"],
)
