# CIFAR-10 Classification

Image classification on the CIFAR-10 dataset using a fully-connected neural network.

## Dataset

CIFAR-10 from [University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html):
- **Training samples**: 50,000 color images (32×32×3 RGB)
- **Test samples**: 10,000 images
- **Classes**: 10 categories
  - 0: airplane
  - 1: automobile
  - 2: bird
  - 3: cat
  - 4: deer
  - 5: dog
  - 6: frog
  - 7: horse
  - 8: ship
  - 9: truck

## Binary Format

Each batch file contains 10,000 records:
- **Record size**: 3073 bytes
  - 1 byte: label (0-9)
  - 3072 bytes: pixel data (1024 red + 1024 green + 1024 blue)
- Training: `data_batch_1.bin` through `data_batch_5.bin`
- Test: `test_batch.bin`

## Architecture

```
3072 (32×32×3 RGB pixels)
  ↓  ReLU
512 neurons
  ↓  ReLU
256 neurons
  ↓  Softmax
 10 classes
```

## Training Configuration

- **Loss**: Cross-Entropy
- **Optimizer**: SGD
- **Learning Rate**: 0.005
- **Batch Size**: 128
- **Epochs**: 20
- **Subset**: 10k train / 2k test (for speed; set to 0 for full dataset)

## Running the Example

```bash
bazel run //examples/cifar:cifar
```

## Expected Performance

**IMPORTANT:** CIFAR-10 is designed for Convolutional Neural Networks (CNNs).

### Typical Results:
- **Fully-Connected Network (this implementation)**: **40-50%**
- **Random guessing**: 10%
- **CNN (ResNet, VGG, etc.)**: 90-95%
- **State-of-the-art (Vision Transformers)**: 99%+

### Why is CIFAR-10 Hard for FC Networks?

1. **Spatial structure**: FC networks ignore the 2D arrangement of pixels
2. **Translation invariance**: An object at different positions looks different to FC
3. **Color complexity**: Natural images with backgrounds, lighting, perspective
4. **Too many parameters**: 3072 inputs → millions of weights without sharing

## Interpretation

If you achieve **40-50% accuracy**, that's **excellent for a fully-connected network**!

- 4-5x better than random guessing
- Competitive with classical FC approaches
- Shows the network is learning features
- Demonstrates the need for CNNs on image data

## Comparison with Fashion-MNIST

| Dataset | Type | FC Accuracy | CNN Accuracy |
|---------|------|-------------|--------------|
| Fashion-MNIST | Grayscale icons | 85-89% | 93-95% |
| CIFAR-10 | RGB photos | **40-50%** | 90-95% |

The gap shows why CNNs revolutionized computer vision!

## Improving Performance

To reach 90%+ on CIFAR-10, you would need:
- Convolutional layers
- Pooling layers
- Batch normalization
- Data augmentation (flips, crops, color jitter)
- Dropout/regularization
- Deeper architectures (ResNet-18, ResNet-50)

This is a future direction for your `nm` library!
