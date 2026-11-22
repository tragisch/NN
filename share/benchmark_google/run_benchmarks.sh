#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_ROOT="${SCRIPT_DIR}/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULTS_ROOT}/${TIMESTAMP}"
HISTORY_DIR="${SCRIPT_DIR}/history"
PLOTS_DIR="${SCRIPT_DIR}/plots"

MODE="${1:-quick}"
MIN_TIME="${2:-0.2s}"
KEEP_RUNS="${KEEP_RUNS:-30}"

mkdir -p "${RUN_DIR}"
mkdir -p "${HISTORY_DIR}"
mkdir -p "${PLOTS_DIR}"

cd "${REPO_ROOT}"

echo "[bench] Building benchmark binaries in opt mode..."
bazel build -c opt \
  //share/benchmark_google:bench_nm_relu \
  //share/benchmark_google:bench_nm_mnist \
  //share/benchmark_google:bench_nm_seq_conv

HOST_NAME="$(hostname)"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
else
  GIT_COMMIT="unknown"
fi

run_case() {
  local bin_target="$1"
  local case_name="$2"
  local filter="$3"

  echo "[bench] ${case_name} (${filter})"

  bazel run -c opt "${bin_target}" -- \
    --benchmark_filter="${filter}" \
    --benchmark_min_time="${MIN_TIME}" \
    --benchmark_out="${RUN_DIR}/${case_name}.json" \
    --benchmark_out_format=json \
    > "${RUN_DIR}/${case_name}.console.txt"

  bazel run -c opt "${bin_target}" -- \
    --benchmark_filter="${filter}" \
    --benchmark_min_time="${MIN_TIME}" \
    --benchmark_out="${RUN_DIR}/${case_name}.csv" \
    --benchmark_out_format=csv \
    > /dev/null
}

if [[ "${MODE}" == "quick" ]]; then
  run_case "//share/benchmark_google:bench_nm_relu" "relu_256x256" "BM_nm_relu/256/256"
  run_case "//share/benchmark_google:bench_nm_mnist" "mnist_predict_256_128" "BM_nm_mnist_predict/256/128"
  run_case "//share/benchmark_google:bench_nm_seq_conv" "seq_conv_predict_auto_nchw" "BM_nm_seq_conv_predict/8/3/32/32/16/3/0/0"
elif [[ "${MODE}" == "full" ]]; then
  run_case "//share/benchmark_google:bench_nm_relu" "relu_256x256" "BM_nm_relu/256/256"
  run_case "//share/benchmark_google:bench_nm_relu" "relu_1024x1024" "BM_nm_relu/1024/1024"
  run_case "//share/benchmark_google:bench_nm_relu" "relu_2048x2048" "BM_nm_relu/2048/2048"

  run_case "//share/benchmark_google:bench_nm_mnist" "mnist_predict_256_128" "BM_nm_mnist_predict/256/128"
  run_case "//share/benchmark_google:bench_nm_mnist" "mnist_predict_1024_128" "BM_nm_mnist_predict/1024/128"
  run_case "//share/benchmark_google:bench_nm_mnist" "mnist_train_epoch_1000_128" "BM_nm_mnist_train_one_epoch/1000/128"

  run_case "//share/benchmark_google:bench_nm_seq_conv" "seq_conv_predict_auto_nchw" "BM_nm_seq_conv_predict/8/3/32/32/16/3/0/0"
  run_case "//share/benchmark_google:bench_nm_seq_conv" "seq_conv_predict_gemm_nchw" "BM_nm_seq_conv_predict/8/3/32/32/16/3/2/0"
else
  echo "Unknown mode '${MODE}'. Use: quick | full"
  exit 2
fi

cat > "${RUN_DIR}/README.txt" <<EOF
Benchmark run completed.

Mode: ${MODE}
Min time: ${MIN_TIME}
Timestamp: ${TIMESTAMP}
Host: $(uname -a)
Git commit: ${GIT_COMMIT}

Files:
$(cd "${RUN_DIR}" && ls -1)
EOF

echo "[bench] Aggregating history..."
python3 "${SCRIPT_DIR}/aggregate_benchmarks.py" \
  --run-dir "${RUN_DIR}" \
  --history-dir "${HISTORY_DIR}" \
  --run-id "${TIMESTAMP}" \
  --mode "${MODE}" \
  --min-time "${MIN_TIME}" \
  --host "${HOST_NAME}" \
  --git-commit "${GIT_COMMIT}"

echo "[bench] Generating trend plots (if matplotlib is available)..."
python3 "${SCRIPT_DIR}/plot_benchmark_trends.py" \
  --history-csv "${HISTORY_DIR}/benchmark_history.csv" \
  --out-dir "${PLOTS_DIR}" || true

echo "[bench] Retention: keeping newest ${KEEP_RUNS} run directories..."
run_dirs=()
while IFS= read -r d; do
  run_dirs+=("${d}")
done < <(find "${RESULTS_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)

run_count="${#run_dirs[@]}"
if (( run_count > KEEP_RUNS )); then
  to_delete=$((run_count - KEEP_RUNS))
  for ((i = 0; i < to_delete; i++)); do
    old_dir="${run_dirs[i]}"
    rm -rf "${old_dir}"
    echo "[bench] Removed old run dir: ${old_dir}"
  done
fi

echo "[bench] Done. Results in: ${RUN_DIR}"
echo "[bench] History: ${HISTORY_DIR}/benchmark_history.csv"
echo "[bench] Plots:   ${PLOTS_DIR}"
