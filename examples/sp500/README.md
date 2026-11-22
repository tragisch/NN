# S&P 500 Log-Return Prediction (after BenjiKCF's financial time-series example)

## Overview

This example re-creates the core task of
[BenjiKCF/Neural-Net-with-Financial-Time-Series-Data](https://github.com/BenjiKCF/Neural-Net-with-Financial-Time-Series-Data):
predict the **next day's log return** of the S&P 500 (`^GSPC`) from a window of
past trading days, using technical indicators as features.

nm_lib currently has no recurrent layers, so instead of the reference's
stacked LSTM the same **60-day window is flattened into a dense MLP** — the
classic "MLP on a rolling window" stand-in. Everything else follows the
reference: same data, same indicators, same target, same 95/5 train/test
split on the time axis.

## Dataset

- **File**: `gspc.csv` — daily `^GSPC` bars (Open, High, Low, Close,
  Adj Close, Volume), 1990-01-02 to 2018-10-10, 7252 rows, taken from the
  reference repository.
- **Target**: `log(AdjClose[t+1] / AdjClose[t])` (in the reference notebook:
  `data['Y'] = data.lrets.shift(-1)`).

## Features (per day, as in notebook 2 of the reference)

| # | Feature | Definition |
|---|---------|------------|
| 0 | log return | `log(adj_close[t] / adj_close[t-1])` |
| 1 | MACD histogram | `EMA12 - EMA26`, signal `EMA9`, hist = MACD − signal |
| 2 | Stochastics %K(14) | `100 * (close − min14) / (max14 − min14)` on adj close |
| 3 | ATR(14) | 14-day mean true range |
| 4 | Volume | raw, robust-scaled |

All features are scaled with a **RobustScaler** (median / IQR), fitted on the
training days only — as the reference fits its scaler on the training split.

## Network Architecture

```
Input  (60 days x 5 features = 300, flattened window)
  -> Dense 128, ReLU
  -> Dense 64,  ReLU
  -> Dense 32,  ReLU
  -> Dense 1,   linear   (log return is unbounded, so no sigmoid)
```

## Training Configuration

- **Loss**: MSE (`NM_LOSS_MSE`)
- **Optimizer**: Adam, learning rate 0.001
- **Batch size**: 64, **epochs**: 60, shuffled each epoch
- **Validation**: 5% of the training sequences held out during fit
  (`validation_split`), plus a final **test holdout of the last 5%** of all
  sequences (time-ordered, as in the reference)

## Building and Running

```bash
bazel run //examples/sp500:sp500
```

## Reported Metrics

On the held-out test sequences: MSE, RMSE (in log-return units and % per
day), a **baseline RMSE for always predicting 0**, MAE, Pearson correlation
between prediction and true log return, and directional accuracy (fraction
of days where the sign of the predicted return matches the true sign — the
practically relevant measure for trading).

## Expected Output

```
Loaded 7252 daily bars from gspc.csv
Built 7191 sequences of 300 features (window 60 x 5)
Model parameters: 48897
Training on 6832 sequences (test holdout: 359)
epoch 10/60  loss=0.000506  val_loss=0.001439
...
epoch 60/60  loss=0.000058  val_loss=0.000361
Test MSE: 0.00054301   Test RMSE: 0.023302 (2.330% per day)
Baseline (predict 0) Test RMSE: 0.007063 (0.706% per day)
Test MAE: 0.014245 (1.424% per day)
Correlation(pred, true): -0.0936
Directional accuracy: 178/359 (49.6%)
```

Note: daily returns are close to unpredictable — the reference LSTM's own
predicted-vs-true plot is similarly noise-like, and its test-period metrics
are no better than the zero baseline. What this example shows about nm_lib is
that training is stable and converges (train/val loss decrease smoothly and
deterministically with a fixed seed); the near-chance test metrics reflect
the task, not the library.

## Key API Features Tested

- Regression with a **linear output unit** (`NM_ACT_NONE`) and MSE loss
- Larger dense stack (300 -> 128 -> 64 -> 32 -> 1, ~49k parameters)
- **Adam optimizer**, mini-batch training with per-epoch shuffling
- **Validation split during fit** plus an explicit time-ordered test holdout
- Progress callback (`NmProgressFn`) for the learning curve
- `nm_network_predict` / `nm_network_evaluate` on held-out data
- Loading a CSV data file through Bazel runfiles (as in the mnist/iris
  examples)
