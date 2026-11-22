load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

cc_binary(
    name = "graphir_export",
    srcs = ["main.c"],
    deps = [
        "//app/nm_lib:nm_engine_c",
        "//app/nm_lib:nm_export_graphir",
    ],
    visibility = ["//visibility:public"],
)
