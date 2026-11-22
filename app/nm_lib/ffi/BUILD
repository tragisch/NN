# Self-contained static archive of the user API (nm_api.h) for foreign
# callers such as GraphIR's Rust adapter. It bundles nm_lib and its C
# dependencies; the caller links the system frameworks itself (see the
# linkopts of //app/nm_lib:nm_legacy_core).
#
#   bazel build //app/nm_lib/ffi:nm_api_static
#   -> bazel-bin/app/nm_lib/ffi/libnm_api_static.a

load("@rules_cc//cc:cc_static_library.bzl", "cc_static_library")

cc_static_library(
    name = "nm_api_static",
    visibility = ["//visibility:public"],
    deps = ["//app/nm_lib:nm_api"],
)
