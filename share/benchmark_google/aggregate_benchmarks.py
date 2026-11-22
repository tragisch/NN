#!/usr/bin/env python3
"""Aggregate Google Benchmark JSON results into a compact history file.

This script scans a single run directory (timestamp folder) and appends one row
per benchmark result to a central CSV and JSONL history.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate benchmark run results")
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory with *.json files")
    parser.add_argument("--history-dir", required=True, type=Path, help="Directory for history files")
    parser.add_argument("--run-id", required=True, help="Timestamp-like run id")
    parser.add_argument("--mode", default="quick", help="Run mode label")
    parser.add_argument("--min-time", default="0.2s", help="Configured benchmark min time")
    parser.add_argument("--host", default="", help="Host name")
    parser.add_argument("--git-commit", default="unknown", help="Git commit SHA")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_record(
    *,
    payload: Dict[str, Any],
    source_file: str,
    run_id: str,
    mode: str,
    min_time: str,
    host: str,
    git_commit: str,
) -> List[Dict[str, Any]]:
    context = payload.get("context", {})
    bench_list = payload.get("benchmarks", [])
    created_at = datetime.now(timezone.utc).isoformat()

    records: List[Dict[str, Any]] = []
    for bench in bench_list:
        if bench.get("run_type") not in (None, "iteration"):
            continue

        record = {
            "created_at_utc": created_at,
            "run_id": run_id,
            "mode": mode,
            "min_time": min_time,
            "host": host or context.get("host_name", ""),
            "git_commit": git_commit,
            "source_file": source_file,
            "benchmark_name": bench.get("name", ""),
            "label": bench.get("label", ""),
            "time_unit": bench.get("time_unit", ""),
            "cpu_time": bench.get("cpu_time", ""),
            "real_time": bench.get("real_time", ""),
            "iterations": bench.get("iterations", ""),
            "bytes_per_second": bench.get("bytes_per_second", ""),
            "items_per_second": bench.get("items_per_second", ""),
            "num_cpus": context.get("num_cpus", ""),
            "mhz_per_cpu": context.get("mhz_per_cpu", ""),
            "library_version": context.get("library_version", ""),
            "library_build_type": context.get("library_build_type", ""),
        }
        records.append(record)
    return records


def append_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    history_dir = args.history_dir

    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    json_files = sorted(run_dir.glob("*.json"))
    if not json_files:
        print(f"[aggregate] No JSON files found in {run_dir}")
        return 0

    all_rows: List[Dict[str, Any]] = []
    for path in json_files:
        payload = load_json(path)
        rows = normalize_record(
            payload=payload,
            source_file=path.name,
            run_id=args.run_id,
            mode=args.mode,
            min_time=args.min_time,
            host=args.host,
            git_commit=args.git_commit,
        )
        all_rows.extend(rows)

    csv_path = history_dir / "benchmark_history.csv"
    jsonl_path = history_dir / "benchmark_history.jsonl"
    append_csv(csv_path, all_rows)
    append_jsonl(jsonl_path, all_rows)

    print(f"[aggregate] Added {len(all_rows)} row(s)")
    print(f"[aggregate] CSV:   {csv_path}")
    print(f"[aggregate] JSONL: {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
