# Moons — make_moons Binary Classification

Vergleichsbeispiel zum PyTorch-Referenz-Repo
[IinjyI/Hands-On-Neural-Networks](https://github.com/IinjyI/Hands-On-Neural-Networks),
**Lab-08 (Dense Neural Networks)**: Dort wird `sklearn.datasets.make_moons`
mit einem MLP klassifiziert.

## Datensatz

- `data/moons.csv`: 2000 Samples, 2 Features, 2 Klassen (balanciert 1000/1000)
- Erzeugt exakt wie `make_moons(n_samples=2000, noise=0.2, random_state=42)`
  (sklearn-Algorithmus 1:1 nachimplementiert, siehe `data/make_moons.py`)
- Format: `x1,x2,label` pro Zeile; das Dataset ist wie in sklearn permutiert,
  die letzten 20 % dienen als Testset

## Modell

MLP wie im Referenz-Lab: `2 -> 16 (ReLU) -> 16 (ReLU) -> 2 (Softmax)`,
Cross-Entropy, Batch 32, lr=0.05, 500 Epochen, Seed 42.
Features werden per Min-Max auf [0, 1] normalisiert.

## Ausführen

```bash
bazel run //examples/moons:moons
```

## Vergleich mit der PyTorch-Referenz (Lab-08)

| | nm_lib (C) | PyTorch (Lab-08, make_moons noise=0.2) |
|---|---|---|
| Test-Genauigkeit | **97,00 %** (Train 97,00 %, Loss 0.080) | typischerweise ~97–99 % |
| Architektur | 2-16-16-2 MLP | 2-16-16-1 MLP |

Die sklearn-Moons mit noise=0.2 sind mit einem kleinen MLP nicht linear
trennbar; beide Implementierungen sollten eine Test-Genauigkeit im hohen
90-%-Bereich erreichen.
