"""PyTorch side of the comparison with nm_lib (see bench_nm.c).

Same models, data and measurements; prints one JSON object per line.
usage: python3 bench_torch.py [threads]   (threads defaults to PyTorch's own)
"""
import json
import os
import statistics
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # duplicate OpenMP runtimes on macOS
import torch
from torch import nn

BATCH, WARMUP, RUNS = 64, 30, 200

if len(sys.argv) > 1:
    torch.set_num_threads(int(sys.argv[1]))
TAG = f"torch{torch.get_num_threads()}"


def fill(rows, cols):
    i = torch.arange(rows * cols, dtype=torch.int64)
    return (((i * 7) % 13).float() / 13.0).reshape(rows, cols)


def mlp():
    torch.manual_seed(1)
    return nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(),
                         nn.Linear(64, 10))


def cnn():
    torch.manual_seed(1)
    return nn.Sequential(nn.Unflatten(1, (1, 28, 28)), nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                         nn.MaxPool2d(2), nn.Flatten(), nn.Linear(8 * 14 * 14, 10))


def timed(fn):
    samples = []
    for i in range(WARMUP + RUNS):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        if i >= WARMUP:
            samples.append((t1 - t0) / 1e3)
    samples.sort()
    return samples[len(samples) // 2], samples[len(samples) * 9 // 10]


def run(label, build):
    model = build()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    x = fill(BATCH, 784)
    y = torch.arange(BATCH) % 10

    def step():
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    median, p90 = timed(step)
    print(json.dumps({"name": f"{label}_train_step_b{BATCH}", "median_us": round(median, 1),
                      "p90_us": round(p90, 1)}))
    model.eval()
    with torch.no_grad():
        for rows in (1, BATCH):
            inp = fill(rows, 784)
            median, p90 = timed(lambda: model(inp).softmax(1))
            print(json.dumps({"name": f"{label}_infer_b{rows}", "median_us": round(median, 1),
                              "p90_us": round(p90, 1)}))


run("mlp", mlp)
run("cnn", cnn)
