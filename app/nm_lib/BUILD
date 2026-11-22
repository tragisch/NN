load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

# define the platforms
constraint_setting(name = "custom_os")
constraint_value(
    name = "macos_arm64",
    constraint_setting = ":custom_os",
)

platform(
    name = "macos_arm64_platform",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
        ":macos_arm64",
    ],
)
config_setting(
    name = "on_macos_arm64",
    constraint_values = [":macos_arm64"],
)

# ── Build configurations ──────────────────────────────────────────
#
#   bazel build --config=dev   → warnings + sanitizers, no optimisation
#   bazel build --config=prod  → optimised, benchmarks  (default via .bazelrc)
#
# The config_setting below is toggled by --define=NM_DEV=1 (see .bazelrc).

config_setting(
    name = "nm_dev_build",
    values = {"define": "NM_DEV=1"},
)

# Shared warning flags — always on
nm_warn_copts = [
    "-Wall",
    "-Wextra",
    "-Wshadow",
    "-Wconversion",
    "-Wno-sign-conversion",
]

# Development: debug info + sanitizers, no optimisation
nm_dev_copts = nm_warn_copts + [
    "-O0",
    "-g",
    "-Wno-macro-redefined",  # Bazel toolchain injects _FORTIFY_SOURCE
    "-fsanitize=address",
    "-fsanitize=undefined",
    "-fno-omit-frame-pointer",
    "-Xpreprocessor",
]

nm_dev_linkopts = [
    "-fsanitize=address",
    "-fsanitize=undefined",
]

# Production: aggressive optimisation, safe fast-math subset
nm_prod_copts = nm_warn_copts + [
    "-O3",
    "-fno-math-errno",
    "-fno-signed-zeros",
    "-fno-trapping-math",
    "-Xpreprocessor",
]

default_deps = [
    "@openmp//:libomp",
    "@pcg",
    "//third_party/log.c:log",
]

# build configurations for different matrix libraries
config_setting(
    name = "use_accelerate",
    values = {"define": "USE_ACCELERATE=1"},
)

config_setting(
    name = "use_openblas",
    values = {"define": "USE_OPENBLAS=1"},
)

config_setting(
    name = "use_accelerate_mps",
    values = {"define": "USE_ACCELERATE_MPS=1"},
)

################# Libraries #################
# Load the matrix library
#############################################


cc_library(
    name = "nm_legacy_core",
    srcs = [
        "src/nm_graph_ir.c",
        "src/nm_frontend_internal.h",
        "src/nm_graph_interp.c",
        "src/nm_graph_runtime.c",
        "src/nm_graph_ir_internal.h",
        "src/nm.c",
        "src/nm_model.c",
        "src/nm_model_core.c",
        "src/nm_model_exec.c",
        "src/nm_model_internal.h",
        "src/nm_model_types_internal.h",
        "src/nm_model_io.c",
        "src/nm_model_layers.c",
        "src/nm_model_lowering.c",
        "src/nm_model_runtime.c",
        "src/nm_model_serialization.h",
    ],
    hdrs = [
        "include/nm.h",
        "include/nm_input.h",
        "include/nm_types.h",
    ],
    copts = select({
        ":nm_dev_build": nm_dev_copts,
        "//conditions:default": nm_prod_copts,
    }) + ["-fopenmp"] + select({
        ":on_macos_arm64": [
            "-DUSE_ACCELERATE_MPS",
            "-DACCELERATE_NEW_LAPACK",
            "-DUSE_MPS",
            "-fobjc-arc",
        ],
        "//conditions:default": [],
    }) + select({
        ":use_openblas": [
            "-DUSE_OPENBLAS",
        ],
        "//conditions:default": [],
    }),
    includes = [
        "include",
        "src",
    ],
    linkopts = select({
        ":nm_dev_build": nm_dev_linkopts,
        "//conditions:default": [],
    }) + select({
        ":on_macos_arm64": [
            "-framework Accelerate",
            "-framework Metal",
            "-framework MetalPerformanceShaders",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = default_deps + [
        "@mmatrix//app/matrix:sm",
        "@mmatrix//app/tensor:st",
        "@mmatrix//app/vector:sv_optim",
        "@xxhash//:xxhash",
        "@zstd//:zstd",
    ] + select({
        ":use_openblas": ["@openblas"],
        "//conditions:default": [],
    })
)

alias(
    name = "nm",
    actual = ":nm_legacy_core",
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "export_graphir_fixture",
    srcs = ["tests/export_graphir_fixture.c"],
    deps = [
        ":nm_engine_c",
        ":nm_export_graphir",
    ],
)

cc_binary(
    name = "benchmark_deepmodel",
    srcs = ["tests/benchmark_deepmodel.c"],
    deps = [
        ":nm_engine_c",
        ":nm_frontend",
    ],
)


cc_library(
    name = "nm_capabilities",
    srcs = ["src/nm_capabilities.c"],
    hdrs = ["include/nm_capabilities.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "print_capabilities",
    srcs = ["tools/print_capabilities.c"],
    deps = [":nm_capabilities"],
)

cc_library(
    name = "nm_frontend",
    srcs = ["src/nm_frontend.c"],
    hdrs = [
        "include/nm_frontend.h",
        "include/nm_types.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [":nm_legacy_core"],
)

cc_library(
    name = "nm_halfprec",
    hdrs = ["src/nm_halfprec.h"],
    includes = ["src"],
)

cc_library(
    name = "nm_import_graphir",
    srcs = [
        "src/nm_girw.c",
        "src/nm_import_graphir.c",
    ],
    hdrs = [
        "include/nm_girw.h",
        "include/nm_import_graphir.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":nm_frontend",
        ":nm_halfprec",
        ":nm_legacy_core",
        "@xxhash//:xxhash",
    ],
)

cc_library(
    name = "nm_weights",
    srcs = ["src/nm_weightset.c"],
    hdrs = ["include/nm_weightset.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":nm_import_graphir",
        ":nm_legacy_core",
        "@xxhash//:xxhash",
    ],
)

cc_library(
    name = "nm_connector",
    srcs = [
        "src/nm_connector.c",
        "src/nm_transport_file.c",
        "src/nm_transport_memory.c",
    ],
    hdrs = ["include/nm_connector.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":nm_engine_c",
        ":nm_weights",
    ],
)


cc_library(
    name = "nm_engine_c",
    srcs = ["src/nm_engine_c.c"],
    hdrs = ["include/nm_engine_c.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":nm_frontend",
        ":nm_import_graphir",
        ":nm_legacy_core",
        ":nm_weights",
    ],
)

cc_library(
    name = "nm_api",
    srcs = ["src/nm_api.c"],
    hdrs = [
        "include/nm_api.h",
        "include/module.modulemap",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":nm_engine_c",
        ":nm_export_graphir",
        ":nm_import_graphir",
        ":nm_legacy_core",
        ":nm_weights",
    ],
)

cc_library(
    name = "nm_export_graphir",
    srcs = [
        "src/nm_export_graphir.c",
    ],
    hdrs = ["include/nm_export_graphir.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":nm_frontend",
        ":nm_halfprec",
        ":nm_legacy_core",
        ":nm_import_graphir",
    ],
)

cc_library(
    name = "nm_frontend_legacy_import",
    srcs = ["src/nm_frontend_legacy_import.c"],
    visibility = ["//visibility:public"],
    deps = [
        ":nm_frontend",
        ":nm_legacy_core",
    ],
)


##############################################
# Tests
#############################################

load("//:tools/unity_test.bzl", "unity_test")

## Unit tests:

unity_test(
    name = "test_nm",
    srcs = ["tests/test_nm.c"],
    my_config = ["tests/my_config.yml"],
    deps = [":nm"],
    tags = ["smoke"],
)

unity_test(
    name = "test_nm_graphir_export",
    srcs = ["tests/test_nm_graphir_export.c"],
    my_config = ["tests/my_config.yml"],
    data = ["tests/fixtures/weights_v2.girw"],
    deps = [
        "//app/nm_lib:nm_engine_c",
        "//app/nm_lib:nm_export_graphir",
        "//app/nm_lib:nm_frontend_legacy_import",
        "//app/nm_lib:nm_import_graphir",
    ],
    tags = ["smoke"],
)

unity_test(
    name = "test_nm_weightset",
    srcs = ["tests/test_nm_weightset.c"],
    my_config = ["tests/my_config.yml"],
    deps = [
        "//app/nm_lib:nm_engine_c",
        "//app/nm_lib:nm_export_graphir",
        "//app/nm_lib:nm_import_graphir",
        "//app/nm_lib:nm_weights",
    ],
    tags = ["smoke"],
)

unity_test(
    name = "test_nm_connector",
    srcs = ["tests/test_nm_connector.c"],
    my_config = ["tests/my_config.yml"],
    deps = [
        "//app/nm_lib:nm_connector",
        "//app/nm_lib:nm_export_graphir",
        "//app/nm_lib:nm_import_graphir",
    ],
    tags = ["smoke"],
)

unity_test(
    name = "test_nm_conformance",
    srcs = ["tests/test_nm_conformance.c"],
    my_config = ["tests/my_config.yml"],
    data = glob(["tests/fixtures/conformance/*"]) + ["tests/fixtures/nm_lib.profile.json"],
    deps = [
        "//app/nm_lib:nm_capabilities",
        "//app/nm_lib:nm_engine_c",
        "//app/nm_lib:nm_import_graphir",
    ],
    tags = ["smoke"],
)

unity_test(
    name = "test_nm_interp_conv",
    srcs = ["tests/test_nm_interp_conv.c"],
    my_config = ["tests/my_config.yml"],
    data = glob(["tests/fixtures/interp/*"]) + glob(["tests/fixtures/conformance/*"]),
    deps = [
        "//app/nm_lib:nm_engine_c",
        "//app/nm_lib:nm_import_graphir",
    ],
    tags = ["smoke"],
)

unity_test(
    name = "test_nm_halfprec",
    srcs = ["tests/test_nm_halfprec.c"],
    my_config = ["tests/my_config.yml"],
    deps = ["//app/nm_lib:nm_halfprec"],
    tags = ["smoke"],
)

unity_test(
    name = "test_nm_api",
    srcs = ["tests/test_nm_api.c"],
    my_config = ["tests/my_config.yml"],
    data = glob(["tests/fixtures/conformance/*"]),
    deps = ["//app/nm_lib:nm_api"],
    tags = ["smoke"],
)

unity_test(
    name = "test_nm_gradients",
    srcs = ["tests/test_nm_gradients.c"],
    my_config = ["tests/my_config.yml"],
    deps = ["//app/nm_lib:nm_engine_c"],
    tags = ["smoke"],
)

unity_test(
    name = "test_nm_girw",
    srcs = ["tests/test_nm_girw.c"],
    my_config = ["tests/my_config.yml"],
    data = ["tests/fixtures/weights_v2.girw"],
    deps = ["//app/nm_lib:nm_import_graphir"],
    tags = ["smoke"],
)

# Benchmarks against PyTorch: python3 app/nm_lib/benchmarks/compare.py
cc_binary(
    name = "bench_nm",
    srcs = ["benchmarks/bench_nm.c"],
    deps = [
        ":nm_engine_c",
        ":nm_frontend",
    ],
)
