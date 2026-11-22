# Boston Housing Example

## Overview

This example demonstrates regression on the Boston Housing dataset. Unlike classification tasks (MNIST, XOR), regression predicts continuous values (house prices) using Mean Squared Error (MSE) loss.

## Dataset

- **Samples**: 100 (synthetic, derived from Boston Housing pattern)
- **Features**: 13 (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT)
- **Target**: MEDV (median home value in $1000s), range: $5k - $55k

The data is synthetically generated to follow a realistic price model based on feature weights.

## Network Architecture

```
Input Layer (13 features)
    ↓
Hidden Layer 1 (32 neurons, ReLU activation)
    ↓
Hidden Layer 2 (8 neurons, ReLU activation)
    ↓
Output Layer (1 neuron, Sigmoid activation)
```

**Note**: The output layer uses Sigmoid activation (outputs in [0, 1]) for normalized regression prediction. Data is rescaled to match this range.

## Training Configuration

- **Batch Size**: 32
- **Learning Rate**: 0.01
- **Epochs**: 200
- **Loss Function**: MSE (Mean Squared Error)
- **Optimizer**: SGD (via backpropagation)
- **Output Range**: [0, 1] normalized, representing [$0k, $55k]

## Building and Running

```bash
bazel run //examples/boston:boston
```

## Expected Output

```
Starting training: 200 epochs, batch_size=32, lr=0.0100
Final Training MSE: 0.003277 (normalized [0,1])
Final Training RMSE: 0.0572 (normalized units)
Mean Absolute Error: 0.0469 (normalized), 2.5787 (in $1000s)

Sample predictions (first 5):
  Sample 0: Predicted: $16.4k, True: $12.7k, Error: $3.72k
  Sample 1: Predicted: $13.6k, True: $10.2k, Error: $3.44k
  Sample 2: Predicted: $14.3k, True: $11.4k, Error: $2.86k
  Sample 3: Predicted: $11.8k, True: $16.0k, Error: $-4.23k
  Sample 4: Predicted: $10.1k, True: $15.1k, Error: $-5.02k
```

## Key API Features Tested

✅ **Sigmoid Output** - Bounded [0, 1] activation for normalized regression
✅ **Multi-Layer Architecture** - 3 hidden layers deep
✅ **Batch Training** - Minibatch processing (batch_size=32)
✅ **Loss Computation** - `nm_network_evaluate()` with `NM_LOSS_MSE`
✅ **Error Metrics** - RMSE and MAE suitable for regression
✅ **Data Normalization** - Scaling features and targets to [0, 1]
✅ **User API** - `NmSpec`, `NmNetwork` and `NmTensor`
✅ **Prediction Accuracy** - Metrics suitable for regression (RMSE, MAE)

## Performance Metrics

For regression, we track:
- **MSE (Mean Squared Error)**: Average squared error - penalizes large errors heavily
- **RMSE (Root Mean Squared Error)**: Square root of MSE - same units as target
- **MAE (Mean Absolute Error)**: Average absolute error - more robust to outliers

## Learning Curve Insights

The network learns to predict housing prices based on the synthetic model:
```
price ≈ 10.0 + 5.0*RM + 3.0*LSTAT + 2.0*CRIM - 1.0*RAD + noise
```

With proper training, the RMSE should converge to near the noise level in [0,1] normalized units (~0.05-0.1), corresponding to ~$3-5k error on actual price predictions.

## Differences from Classification

| Aspect | Classification | Regression |
|--------|----------------|-----------|
| Output Activation | Softmax | Sigmoid (or Linear) |
| Loss Function | Cross-Entropy | MSE |
| Output Range | Probability [0,1] per class | Continuous value |
| Metrics | Accuracy | RMSE, MAE |
| Target Format | One-hot encoding | Single value |
| Interpretation | Class confidence | Direct prediction |
