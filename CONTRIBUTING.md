# Contributing to nm

Thank you for your interest in contributing. This project is a compact C
neural-network library built with Bazel. Contributions are welcome — bug
reports, fixes, documentation, and new examples.

## Getting started

```bash
git clone https://github.com/tragisch/14_NN.git nm
cd nm
bazel build //...          # build all targets
bazel test //app/nm_lib/...  # run library tests
```

Requires: macOS on Apple Silicon, Bazel 10 (see `.bazelversion`), a C compiler
toolchain (Clang/LLVM via `toolchains_llvm`).

## Workflow

1. Fork the repository and create a feature branch.
2. Make your changes. Keep diffs minimal and focused.
3. Run `bazel build //...` and `bazel test //app/nm_lib/...` before pushing.
4. Open a pull request with a clear description of what and why.

## Code style

- C code follows the existing style in `app/nm_lib/src/`.
- Public API changes go in `app/nm_lib/include/nm_api.h` (or the relevant
  header) and must be documented.
- New features should include tests under `app/nm_lib/tests/`.

## Reporting issues

Use GitHub Issues. Include: Bazel version, macOS version, steps to reproduce,
and the full error output.
