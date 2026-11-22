"""This module provides a "helper2 to build a C binary for a matrix."""
load("@rules_cc//cc:defs.bzl", "cc_binary")

def matrix_cc_binary(name, srcs, deps = [], data = [], visibility = None):
    cc_binary(
        name = name,
        srcs = srcs,
        deps = deps,
        data = data,
        linkopts = ["-lc++abi"],
        visibility = visibility,    )

