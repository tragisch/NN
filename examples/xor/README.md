# XOR Example

## Overview

This example demonstrates a simple non-linearly separable binary classification task using a feedforward neural network. The XOR function is a classic problem that requires at least one hidden layer with non-linear activations to solve.

## What is XOR?

XOR (exclusive OR) is a binary classification problem where:
- Input `(0, 0)` → Class 0
- Input `(0, 1)` → Class 1  
- Input `(1, 0)` → Class 1
- Input `(1, 1)` → Class 0

This patterns cannot be separated by a linear decision boundary, so a linear classifier will fail.

## Architecture

```
Input Layer (2 neurons)
    ↓
Hidden Layer (8 neurons, ReLU activation)
    ↓
Output Layer (2 neurons, Softmax activation)
```

## Training Configuration

- **Dataset**: 4 samples (full enumeration of XOR)
- **Batch Size**: 4 (full batch)
- **Learning Rate**: 0.5
- **Epochs**: 5000
- **Loss Function**: Cross-Entropy
- **Optimizer**: SGD (via backpropagation)

## Building and Running

```bash
bazel run //examples/xor:xor
```

## Expected Output

The network should converge to **100% accuracy**:

```
Final Training Accuracy: 100.00%
Detailed predictions:
  XOR(0, 0) -> Predicted: class 0, True: class 0 ✓
  XOR(0, 1) -> Predicted: class 1, True: class 1 ✓
  XOR(1, 0) -> Predicted: class 1, True: class 1 ✓
  XOR(1, 1) -> Predicted: class 0, True: class 0 ✓
```

## Key API Features Tested

✅ **Network Creation** - `nm_spec_new()` and `nm_network_build()`
✅ **Training** - `nm_network_fit()` with `NmFitOptions`
✅ **Prediction** - `nm_network_predict()` inference
✅ **Metrics** - `nm_network_evaluate()` returns `NmMetrics`
✅ **Activations** - ReLU (hidden) and Softmax (output)
✅ **Loss Functions** - Cross-Entropy with softmax shortcut
✅ **Tensor Views** - Pass the input and target arrays without copying

## Performance

- **Compilation Time**: < 1 second
- **Execution Time**: < 0.5 seconds  
- **Memory**: Minimal (4 samples)

This example is perfect for:
- Quick sanity checks after API changes
- Verifying non-linear function learning capability
- Testing on minimal compute resources
- Demonstrating API usage patterns clearly
