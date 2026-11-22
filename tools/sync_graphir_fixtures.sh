#!/bin/bash
# Keeps nm_lib's copies of GraphIR's conformance corpus and nm_lib's published
# capability profile in step. GraphIR owns the corpus; nm_lib owns the profile.
#   sync_graphir_fixtures.sh pull   corpus from GraphIR into nm_lib
#   sync_graphir_fixtures.sh push   nm_lib's profile into GraphIR
# usage: sync_graphir_fixtures.sh pull|push [path-to-GraphIR]
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
GRAPHIR="${2:-$HOME/Documents/Programmierung/Rust/03_Projects/GraphIR}"
case "$1" in
  pull)
    rm -rf "$HERE/app/nm_lib/tests/fixtures/conformance"
    mkdir -p "$HERE/app/nm_lib/tests/fixtures/conformance"
    cp "$GRAPHIR"/tests/fixtures/conformance/* "$HERE/app/nm_lib/tests/fixtures/conformance/"
    echo "corpus copied from $GRAPHIR" ;;
  push)
    (cd "$HERE" && bazel build //app/nm_lib:print_capabilities)
    "$HERE/bazel-bin/app/nm_lib/print_capabilities" > "$GRAPHIR/src/backends/nm_lib.profile.json"
    cp "$GRAPHIR/src/backends/nm_lib.profile.json" "$HERE/app/nm_lib/tests/fixtures/nm_lib.profile.json"
    echo "profile written to $GRAPHIR/src/backends/nm_lib.profile.json" ;;
  *) echo "usage: $0 pull|push [path-to-GraphIR]"; exit 2 ;;
esac
