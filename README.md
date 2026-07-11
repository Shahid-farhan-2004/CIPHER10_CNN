# CIFAR-10 Image Classification using Convolutional Neural Network (CNN) in PyTorch

## Overview

This project demonstrates how to build and train a **Convolutional Neural Network (CNN)** using **PyTorch** to classify color images from the **CIFAR-10** dataset.

The model learns visual features such as edges, textures, and shapes using convolutional layers before classifying images into one of the ten object categories.

---

# Technologies Used

- Python
- PyTorch
- Torchvision

---

# Dataset

The project uses the **CIFAR-10** dataset.

### Dataset Information

- 60,000 color images
- 50,000 training images
- 10,000 testing images
- 10 classes
- Image size: **32 × 32**
- RGB images (3 channels)

---

# CIFAR-10 Classes

| Label | Class |
|--------|-------|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

---

# Project Workflow

```
CIFAR-10 Images
        │
        ▼
Transformations
(ToTensor + Normalize)
        │
        ▼
DataLoader
        │
        ▼
CNN
(Conv → ReLU → Pool)
        │
        ▼
Flatten
        │
        ▼
Fully Connected Layer
        │
        ▼
Output Layer
(10 Classes)
        │
        ▼
CrossEntropyLoss
        │
        ▼
Backpropagation
        │
        ▼
Weight Update (Adam)
```

---

# Data Transformations

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
])
```

### `ToTensor()`

Converts images into PyTorch tensors.

Image values

```
0–255
```

become

```
0.0–1.0
```

---

### `Normalize()`

Normalizes each RGB channel.

Formula

```
Normalized Value

=

(Value − Mean)

/

Standard Deviation
```

Here

```
Mean = (0.5,0.5,0.5)

Std = (0.5,0.5,0.5)
```

After normalization

```
0 → -1

0.5 → 0

1 → 1
```

Normalization helps the neural network train faster and more stably.

---

# Loading Dataset

```python
train_data = datasets.CIFAR10(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_data = datasets.CIFAR10(
    root="./data",
    train=False,
    transform=transform,
    download=True
)
```

`download=True` downloads the dataset if it is not already present.

---

# DataLoader

```python
train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=1000
)
```

### Batch Size

```
64
```

The model trains using **64 images at a time**.

### Shuffle

```
shuffle=True
```

Randomly shuffles training images every epoch, helping the model generalize better.

---

# CNN Architecture

```text
Input
(3×32×32)

      │

Conv2D
3 → 32

      │

ReLU

      │

MaxPool

      │

Conv2D
32 → 64

      │

ReLU

      │

MaxPool

      │

Flatten

      │

Linear
4096 → 256

      │

ReLU

      │

Linear
256 → 10
```

---

# Convolution Layers

## First Convolution

```python
self.conv1 = nn.Conv2d(
    3,
    32,
    kernel_size=3,
    padding=1
)
```

### Meaning

- Input Channels = 3 (RGB)
- Output Channels = 32
- Kernel Size = 3×3
- Padding = 1

Output

```
(32,32,32)
```

---

## Second Convolution

```python
self.conv2 = nn.Conv2d(
    32,
    64,
    kernel_size=3,
    padding=1
)
```

Output

```
(64,16,16)
```

before pooling.

---

# ReLU Activation

```python
F.relu()
```

ReLU replaces all negative values with zero.

Formula

```
ReLU(x)

=

max(0,x)
```

Example

```
Input

[-4,2,-1,5]

↓

Output

[0,2,0,5]
```

Benefits

- Faster learning
- Prevents vanishing gradients
- Adds non-linearity

---

# Max Pooling

```python
self.pool = nn.MaxPool2d(2,2)
```

Reduces image size while keeping important features.

Example

```
4×4

↓

2×2
```

---

# Image Size Through the Network

```
Input

3×32×32

↓

Conv1

32×32×32

↓

Pool

32×16×16

↓

Conv2

64×16×16

↓

Pool

64×8×8

↓

Flatten

4096
```

---

# Flatten Layer

```python
x = torch.flatten(x,1)
```

Converts

```
(batch,64,8,8)
```

into

```
(batch,4096)
```

Example

```
(64,64,8,8)

↓

(64,4096)
```

The first dimension (batch size) remains unchanged.

---

# Fully Connected Layers

```python
self.fc1 = nn.Linear(
    64*8*8,
    256
)

self.out = nn.Linear(
    256,
    10
)
```

The first layer learns high-level image features.

The second layer produces one score for each CIFAR-10 class.

---

# Forward Pass

```python
x = self.pool(F.relu(self.conv1(x)))
x = self.pool(F.relu(self.conv2(x)))
x = torch.flatten(x,1)
x = F.relu(self.fc1(x))
return self.out(x)
```

The output consists of **10 logits**.

No Softmax is applied because `CrossEntropyLoss` handles it internally.

---

# Loss Function

```python
criterion = nn.CrossEntropyLoss()
```

CrossEntropyLoss compares

- Predicted logits
- Actual class labels

The objective is to minimize this loss during training.

---

# Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Adam updates the network weights using gradients computed during backpropagation.

Advantages

- Fast convergence
- Adaptive learning rate
- Stable optimization

---

# Training Loop

```python
for epoch in range(5):

    for images, labels in train_loader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
```

Each iteration performs:

1. Forward pass
2. Loss computation
3. Clear previous gradients
4. Backpropagation
5. Weight update

---

# Testing

```python
with torch.no_grad():

    outputs = model(images)

    _, predictions = torch.max(outputs,1)
```

`torch.no_grad()` disables gradient computation during evaluation, reducing memory usage and improving speed.

`torch.max(outputs, 1)` selects the class with the highest score.

---

# Accuracy Calculation

```python
correct += (predictions == labels).sum().item()

accuracy = correct / total
```

Formula

```
Accuracy

=

Correct Predictions

/

Total Predictions
```

---

# Shape Flow

```
Batch

(64,3,32,32)

↓

Conv1

(64,32,32,32)

↓

Pool

(64,32,16,16)

↓

Conv2

(64,64,16,16)

↓

Pool

(64,64,8,8)

↓

Flatten

(64,4096)

↓

FC1

(64,256)

↓

Output

(64,10)
```

---

# Important Concepts Learned

- CIFAR-10 dataset
- RGB images
- Image normalization
- DataLoader
- Batch training
- Convolutional layers
- Feature maps
- ReLU activation
- Max Pooling
- Flattening
- Fully Connected Layers
- CrossEntropyLoss
- Adam Optimizer
- Forward propagation
- Backpropagation
- Image classification

---

# Expected Performance

A simple CNN like this typically achieves:

- **60–75% accuracy** after about 5 epochs.
- Higher accuracy can be achieved by training for more epochs or using a deeper architecture.

---

# Notes

- `CrossEntropyLoss` expects **raw logits**, so do **not** apply `Softmax` in the model's final layer.
- `torch.flatten(x, 1)` keeps the batch dimension intact while flattening the remaining dimensions.
- Setting `shuffle=True` for the training loader helps the model learn more effectively by presenting the data in a different order each epoch.

---

# License

This project is intended for educational purposes to demonstrate image classification using a Convolutional Neural Network (CNN) in PyTorch with the CIFAR-10 dataset.
