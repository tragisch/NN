# Iris Classification Example

## Overview

This example demonstrates multi-class classification on the famous **Fisher's Iris dataset** from 1936. It's one of the most classic datasets in machine learning literature and provides a realistic test of a neural network's ability to classify real-world biological data.

## Dataset Details

- **Instances**: 150 (50 per class)
- **Features**: 4 continuous measurements
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Classes**: 3 types of Iris plants
  - Iris-setosa (linearly separable from others)
  - Iris-versicolor (not linearly separable from virginica)
  - Iris-virginica (not linearly separable from versicolor)
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris)

## Network Architecture

```
Input Layer (4 features)
    ↓
Hidden Layer (8 neurons, ReLU activation)
    ↓
Output Layer (3 neurons, Softmax activation)
```

## Data Processing

1. **CSV Loading**: Custom loader in `iris_data.c` parses CSV and one-hot encodes labels
2. **Normalization**: Features scaled to [0, 1] range per feature
3. **Stratified Split**: 80/20 train/test split preserving class distribution
   - Every 5th sample → test set (ensures all 3 classes in test)
   - Train: 120 samples (40 per class)
   - Test: 30 samples (10 per class)

## Training Configuration

- **Batch Size**: 16
- **Learning Rate**: 0.05
- **Epochs**: 200
- **Loss Function**: Cross-Entropy with Softmax shortcut
- **Optimizer**: SGD (via backpropagation)

## Building and Running

```bash
bazel run //examples/iris:iris
```

## Expected Output

```
Loaded 150 Iris samples (4 features, 3 classes)
Features normalized to [0, 1] range
Train set: 120 samples
Test set: 30 samples
Starting training: 200 epochs, batch_size=16, lr=0.0500
Training Accuracy: 95.00%
Test Accuracy: 100.00%

Sample test predictions (first 10):
  Sample 0: Predicted=Setosa (0.989), True=Setosa ✓
  Sample 1: Predicted=Setosa (0.980), True=Setosa ✓
  Sample 2: Predicted=Setosa (0.993), True=Setosa ✓
  ...
```

## Key API Features Tested

✅ **CSV Data Loading** - Custom data loader with file I/O
✅ **Multi-class Classification** - 3-way classification task
✅ **Feature Normalization** - Min-max scaling to [0, 1]
✅ **Stratified Splitting** - Balanced train/test distribution
✅ **Cross-Entropy Loss** - With softmax-crossentropy optimization
✅ **Real-world Data** - Actual measurements from botanical research
✅ **High Accuracy** - Achieves near-perfect classification

## Performance Insights

The network successfully learns to distinguish all three Iris species:

- **Iris-setosa**: Perfectly classified (linearly separable)
- **Iris-versicolor**: High accuracy despite non-linear separation
- **Iris-virginica**: High accuracy despite non-linear separation

The test accuracy of **100%** demonstrates that the neural network can learn non-linear decision boundaries to separate versicolor from virginica, which cannot be separated by a linear classifier.

## Data Quality Note

According to the dataset documentation, there are known errors in the original Fisher data:
- Sample 35: 4th feature should be 0.2 (not as published)
- Sample 38: 2nd and 3rd features have errors

These minor inconsistencies don't significantly impact training.

## Comparison with Classical Methods

Fisher's original paper used linear discriminant analysis (LDA). Neural networks offer advantages:
- Can learn non-linear boundaries automatically
- No assumptions about data distribution
- Easily extensible to more complex feature engineering

This example shows that even a simple 2-layer network can match or exceed classical statistical methods on this benchmark dataset.
