#!/usr/bin/env bash
set -euo pipefail

: "${GRAPHIR_ROOT:?Set GRAPHIR_ROOT to the local GraphIR checkout.}"
if [[ ! -f "$GRAPHIR_ROOT/MODULE.bazel" ]]; then
  echo "GRAPHIR_ROOT is not a GraphIR Bazel workspace: $GRAPHIR_ROOT" >&2
  exit 2
fi

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/nm-graphir-export.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT
graph="$work_dir/model.graphir.json"
expected="$work_dir/expected.txt"
generated="$work_dir/model.rs"
runner="$work_dir/runner.rs"

bazel run //app/nm_lib:export_graphir_fixture -- "$graph" "$expected"
(cd "$GRAPHIR_ROOT" && bazel run //:graphir -- check "$graph")
(cd "$GRAPHIR_ROOT" && bazel run //:graphir -- rust "$graph") > "$generated"

cp "$generated" "$runner"
cat >> "$runner" <<'RUST'
fn main() {
    let expected = std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();
    let expected: Vec<f32> = expected.lines().map(|line| line.parse().unwrap()).collect();
    let output = run_nm_frontend(&[0.0, 0.25, -0.5, 1.0, -0.75, 0.5]);
    assert_eq!(output.len(), expected.len());
    for (actual, expected) in output.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
    }
}
RUST
rustc "$runner" -o "$work_dir/runner"
"$work_dir/runner" "$expected"

# Same model with the weights in a checksummed .girw package.
mkdir "$work_dir/external"
bazel run //app/nm_lib:export_graphir_fixture -- \
  "$work_dir/external/model.graphir.json" "$work_dir/external/expected.txt" \
  "$work_dir/external/model.girw"
(cd "$GRAPHIR_ROOT" && bazel run //:graphir -- check "$work_dir/external/model.graphir.json")
(cd "$GRAPHIR_ROOT" && bazel run //:graphir -- rust "$work_dir/external/model.graphir.json" \
  --weights "$work_dir/external") > "$work_dir/external/model.rs"
echo "GraphIR export verified: $graph"
