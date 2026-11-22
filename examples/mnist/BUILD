load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("//:tools/datasets.bzl", "gunzip_file")

# MNIST data is fetched by Bazel via http_file (see MODULE.bazel) and
# decompressed here so runfiles paths stay examples/mnist/<file>.
gunzip_file("mnist_train_images", "mnist_train_images_gz", "train-images-idx3-ubyte.gz", out = "train-images-idx3-ubyte")
gunzip_file("mnist_train_labels", "mnist_train_labels_gz", "train-labels-idx1-ubyte.gz", out = "train-labels-idx1-ubyte")
gunzip_file("mnist_t10k_images", "mnist_t10k_images_gz", "t10k-images-idx3-ubyte.gz", out = "t10k-images-idx3-ubyte")
gunzip_file("mnist_t10k_labels", "mnist_t10k_labels_gz", "t10k-labels-idx1-ubyte.gz", out = "t10k-labels-idx1-ubyte")

filegroup(
    name = "mnist_data_files",
    srcs = [
        ":train-images-idx3-ubyte",
        ":train-labels-idx1-ubyte",
        ":t10k-images-idx3-ubyte",
        ":t10k-labels-idx1-ubyte",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "mnist",
  srcs = ["main.c"],
    data = [":mnist_data_files"],
    deps = [
      "//app/nm_lib:nm_api",
      ":mnist_data",
      "@argtable3//:argtable3",
    ],
  visibility = ["//visibility:public"],
)

cc_binary(
    name = "main_simple",
    srcs = ["main_simple.c"],
    data = [":mnist_data_files"],
    deps = [
      "//app/nm_lib:nm_api",
      ":mnist_data",
    ],
  visibility = ["//visibility:public"],
)

cc_library(
    name = "mnist_data",
  srcs = ["mnist_data.c", "mnist_data.h"],
  hdrs = ["mnist_data.h"],
    includes = ["."],
    deps = [
      "//app/nm_lib:nm_api",
    ],
  visibility = ["//visibility:public"],
)
