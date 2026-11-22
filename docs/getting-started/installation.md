# Installation

`nm_lib` targets macOS on Apple Silicon. The compute kernels come from the
mMatrix repository (`sm`, `st`) and use Accelerate and Metal Performance
Shaders. There is no Linux build yet.

## Prerequisites

- Bazel (through `bazelisk`), Bzlmod enabled
- Xcode command line tools (clang, the Metal and Accelerate frameworks)
- A checkout of mMatrix. `MODULE.bazel` points at it with a
  `local_repository` and an absolute path; edit that path for your machine.
- Homebrew OpenBLAS is only needed for the optional `--define=USE_OPENBLAS=1`
  build.
- Python 3.11 is only needed to build this documentation.

## Option A: inside this Bazel workspace

Add a target next to your sources and depend on `//app/nm_lib:nm_api`:

```python
cc_binary(
    name = "my_program",
    srcs = ["main.c"],
    deps = ["//app/nm_lib:nm_api"],
)
```

```text
bazel run //examples/mnist_api:mnist_api     # the shipped example
bazel build --config=dev //app/nm_lib:nm_api # warnings plus ASan/UBSan
bazel test //app/nm_lib:all                  # unit and conformance tests
```

`.bazelrc` already selects Apple Silicon, `-c opt` and the Accelerate backend.

## Option B: any other build system (static archive)

`//app/nm_lib/ffi:nm_api_static` bundles `nm_lib` and its C dependencies
(mMatrix kernels, xxhash, zstd, OpenMP runtime, `log.c`, PCG) into one archive:

```text
bazel build //app/nm_lib/ffi:nm_api_static
# -> bazel-bin/app/nm_lib/ffi/libnm_api_static.a
```

Compile and link a program against it. Two include directories are needed,
because `nm_api.h` includes `nm_types.h`, which includes `sm.h` from mMatrix:

```text
NM=/path/to/14_NN
MMATRIX=/path/to/mMatrix

clang -O2 -std=c11 my_program.c \
  -I$NM/app/nm_lib/include \
  -I$MMATRIX/app/matrix/include \
  $NM/bazel-bin/app/nm_lib/ffi/libnm_api_static.a \
  -framework Accelerate \
  -framework Metal \
  -framework MetalPerformanceShaders \
  -framework MetalPerformanceShadersGraph \
  -framework Foundation \
  -o my_program
```

All five frameworks are required; without `MetalPerformanceShadersGraph` the
link fails with `Undefined symbols ... _OBJC_CLASS_$_MPSGraph`.

Notes:

- Programs are plain C (C11). C++ works too; the header has `extern "C"`.
- Swift imports `nm_api.h` through `include/module.modulemap`.
- The archive is what GraphIR's Rust adapter links.
- Rebuild the archive after pulling changes; it is not versioned.

## Runtime notes

- Training and prediction use OpenMP; loops only run in parallel above a size
  threshold, so small networks are not slowed down by thread start-up.
  `OMP_NUM_THREADS` is honoured. See [Backends](../performance/backends.md).
- Error details are also written to stderr by the internal logger. The
  message that `nm_last_error()` returns is the one to show to users.

## Build the documentation

```text
bazel build //:mkdocs          # static site in bazel-bin/site/
bazel run //:mkdocs.serve      # local server
bazel run //:requirements.update   # refresh docs/requirements.txt
```
