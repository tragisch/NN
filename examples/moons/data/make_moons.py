#!/usr/bin/env python3
"""Generate the make_moons dataset exactly as scikit-learn does.

Replicates sklearn.datasets.make_moons(n_samples=2000, noise=0.2,
random_state=42) so results are comparable with the PyTorch reference
lab (https://github.com/IinjyI/Hands-On-Neural-Networks, Lab-08).

Writes moons.csv with columns: x1,x2,label
"""
import numpy as np


def make_moons(n_samples=2000, noise=0.2, random_state=42):
    generator = np.random.RandomState(random_state)

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T
    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)]
    )

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    permutation = generator.permutation(n_samples)
    X = X[permutation]
    y = y[permutation]

    return X, y


if __name__ == "__main__":
    X, y = make_moons()
    with open("moons.csv", "w") as f:
        for (x1, x2), label in zip(X, y):
            f.write(f"{x1:.6f},{x2:.6f},{label}\n")
    print(f"wrote moons.csv: {len(y)} samples, {int(y.sum())} class-1 / {int((1 - y).sum())} class-0")
