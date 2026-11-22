#!/usr/bin/env python3
"""
PyTorch MNIST Benchmark - Direct comparison to nm library.

Architecture:
- Input: 784 (28x28)
- Hidden: 128 (ReLU)
- Output: 10 (Softmax)
- Optimizer: SGD
- Loss: CrossEntropy

Usage:
    python3 benchmark_pytorch.py
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import struct
import numpy as np


def load_mnist_images(filepath, num_samples):
    """Load MNIST images from IDX file."""
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8, count=num_samples * rows * cols)
        images = images.reshape(num_samples, rows * cols).astype(np.float32) / 255.0
    return torch.from_numpy(images)


def load_mnist_labels(filepath, num_samples):
    """Load MNIST labels from IDX file."""
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8, count=num_samples)
    return torch.from_numpy(labels).long()


class MNISTNet(nn.Module):
    """MNIST Network matching nm library architecture."""
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
        # Initialize weights similar to nm (Kaiming/He for ReLU)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_and_evaluate(device_name='mps'):
    """Train and evaluate MNIST network."""
    
    # Configuration matching nm
    train_size = 60000
    test_size = 10000
    batch_size = 128
    learning_rate = 0.02
    epochs = 10
    
    print(f"PyTorch MNIST Benchmark")
    print(f"=====================")
    print(f"Device: {device_name}")
    print(f"Training samples: {train_size}")
    print(f"Test samples: {test_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print()
    
    # Set device
    if device_name == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif device_name == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print("Loading MNIST data...")
    X_train = load_mnist_images('examples/mnist/train-images-idx3-ubyte', train_size)
    y_train = load_mnist_labels('examples/mnist/train-labels-idx1-ubyte', train_size)
    X_test = load_mnist_images('examples/mnist/t10k-images-idx3-ubyte', test_size)
    y_test = load_mnist_labels('examples/mnist/t10k-labels-idx1-ubyte', test_size)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training
    print("Training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    train_time = time.time() - start_time
    
    # Evaluation
    print("Evaluating...")
    eval_start = time.time()
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    eval_time = time.time() - eval_start
    total_time = time.time() - start_time
    
    accuracy = 100.0 * correct / total
    
    print()
    print(f"Results:")
    print(f"========")
    print(f"Training time: {train_time:.2f}s")
    print(f"Evaluation time: {eval_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return total_time, accuracy


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run on MPS (Metal Performance Shaders)
    if torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) available!")
        print()
        mps_time, mps_acc = train_and_evaluate('mps')
        print()
        print("-" * 50)
        print()
        
        # Also run on CPU for comparison
        print("Running CPU comparison...")
        print()
        cpu_time, cpu_acc = train_and_evaluate('cpu')
        print()
        print("=" * 50)
        print(f"MPS Speedup vs CPU: {cpu_time/mps_time:.2f}x")
    else:
        print("MPS not available, running on CPU only")
        train_and_evaluate('cpu')
