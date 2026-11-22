# Fashion-MNIST Classification

Fashion-MNIST classification example using a 3-layer neural network.

## Dataset

Fashion-MNIST from [Zalando Research](https://github.com/zalandoresearch/fashion-mnist):
- **Training samples**: 60,000 grayscale images (28×28 pixels)
- **Test samples**: 10,000 images
- **Classes**: 10 fashion categories
  - 0: T-shirt/top
  - 1: Trouser
  - 2: Pullover
  - 3: Dress
  - 4: Coat
  - 5: Sandal
  - 6: Shirt
  - 7: Sneaker
  - 8: Bag
  - 9: Ankle boot

Fashion-MNIST is a drop-in replacement for MNIST, but **more challenging** than handwritten digit recognition.

## Architecture

```
784 (28×28 pixels)
  ↓  ReLU
256 neurons
  ↓  ReLU
128 neurons
  ↓  Softmax
 10 classes
```

## Training Configuration

- **Loss**: Cross-Entropy
- **Optimizer**: SGD
- **Learning Rate**: 0.01
- **Batch Size**: 128
- **Epochs**: 15

## Running the Example

```bash
bazel run //examples/fashion:fashion
```

## Expected Performance

Fashion-MNIST benchmarks (from Zalando Research):
- **Naive Bayes**: ~84%
- **SVM**: ~89%
- **2-layer NN**: ~88%
- **3-layer NN (this implementation)**: ~87-89%
- **CNN**: ~93%

This simple fully-connected network should achieve **~87-89% test accuracy**, which is competitive with traditional ML methods but below state-of-the-art CNNs.

## Notes

- Fashion-MNIST is significantly harder than MNIST (digit MNIST achieves ~99%)
- Some classes are inherently difficult to distinguish (e.g., T-shirt vs. Shirt, Pullover vs. Coat)
- Convolutional networks perform better on this task due to spatial structure
- The dataset uses the same IDX file format as MNIST
