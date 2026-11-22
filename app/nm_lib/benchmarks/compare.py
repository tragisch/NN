#!/usr/bin/env python3
"""Runs nm_lib and PyTorch benchmarks and prints a comparison table.

usage: compare.py [path-to-bench_nm]
  default: bazel-bin/app/nm_lib/bench_nm from the workspace root
"""
import json
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, "..", "..", ".."))
bench_nm = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "bazel-bin/app/nm_lib/bench_nm")


def collect(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return {j["name"]: j["median_us"] for j in map(json.loads, filter(lambda l: l.startswith("{"), out.splitlines()))}


nm = collect([bench_nm])
torch1 = collect([sys.executable, os.path.join(here, "bench_torch.py"), "1"])
torch_all = collect([sys.executable, os.path.join(here, "bench_torch.py")])

print(f"{'case':<24}{'nm_lib us':>12}{'torch 1T us':>14}{'torch 8T us':>14}{'nm vs 1T':>10}{'nm vs 8T':>10}")
for name in nm:
    a, b, c = nm[name], torch1[name], torch_all[name]
    print(f"{name:<24}{a:>12.1f}{b:>14.1f}{c:>14.1f}{b / a:>9.2f}x{c / a:>9.2f}x")
print("(ratio > 1: nm_lib is faster; medians of 200 runs after 30 warm-up runs)")
