load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("//:tools/datasets.bzl", "gunzip_file")

# Fashion-MNIST data is fetched by Bazel via http_file (see MODULE.bazel)
# and decompressed here so runfiles paths stay examples/fashion/data/<file>.
gunzip_file("fashion_train_images", "fashion_train_images_gz", "train-images-idx3-ubyte.gz", out = "data/train-images-idx3-ubyte")
gunzip_file("fashion_train_labels", "fashion_train_labels_gz", "train-labels-idx1-ubyte.gz", out = "data/train-labels-idx1-ubyte")
gunzip_file("fashion_t10k_images", "fashion_t10k_images_gz", "t10k-images-idx3-ubyte.gz", out = "data/t10k-images-idx3-ubyte")
gunzip_file("fashion_t10k_labels", "fashion_t10k_labels_gz", "t10k-labels-idx1-ubyte.gz", out = "data/t10k-labels-idx1-ubyte")

filegroup(
    name = "fashion_data_files",
    srcs = [
        ":data/t10k-images-idx3-ubyte",
        ":data/t10k-labels-idx1-ubyte",
        ":data/train-images-idx3-ubyte",
        ":data/train-labels-idx1-ubyte",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "fashion",
    srcs = ["main.c"],
    data = [":fashion_data_files"],
    deps = [
      "//app/nm_lib:nm_api",
      ":fashion_data",
      "//third_party/log.c:log",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "fashion_data",
    srcs = ["fashion_data.c", "fashion_data.h"],
    hdrs = ["fashion_data.h"],
    includes = ["."],
    deps = [
      "//app/nm_lib:nm_api",
      "//third_party/log.c:log",
    ],
    visibility = ["//visibility:public"],
)
