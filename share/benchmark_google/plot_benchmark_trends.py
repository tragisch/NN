#!/usr/bin/env python3
"""Render compact trend charts from benchmark history."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark trends")
    parser.add_argument("--history-csv", required=True, type=Path, help="Path to benchmark_history.csv")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for PNG plots")
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=["BM_nm_relu/256/256", "BM_nm_mnist_predict/256/128"],
        help="Benchmark names to plot",
    )
    return parser.parse_args()


def read_history(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    args = parse_args()
    if not args.history_csv.exists():
        print(f"[plot] History CSV not found: {args.history_csv}")
        return 0

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed, skip plotting.")
        print("[plot] Install with: pip install matplotlib")
        return 0

    rows = read_history(args.history_csv)
    if not rows:
        print("[plot] History empty, nothing to plot.")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for bench_name in args.benchmarks:
        selected = [r for r in rows if r.get("benchmark_name") == bench_name]
        selected.sort(key=lambda r: r.get("run_id", ""))
        if not selected:
            continue

        x = [r.get("run_id", "") for r in selected]
        y = []
        for r in selected:
            try:
                y.append(float(r.get("cpu_time", "nan")))
            except ValueError:
                y.append(float("nan"))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y, marker="o", linewidth=1.5)
        ax.set_title(bench_name)
        ax.set_xlabel("run_id")
        ax.set_ylabel("cpu_time (ns)")
        ax.grid(True, linestyle="--", alpha=0.35)
        if len(x) > 8:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")

        slug = bench_name.replace("/", "_").replace(" ", "_")
        out_file = args.out_dir / f"trend_{slug}.png"
        fig.tight_layout()
        fig.savefig(out_file, dpi=160)
        plt.close(fig)
        print(f"[plot] Wrote {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
