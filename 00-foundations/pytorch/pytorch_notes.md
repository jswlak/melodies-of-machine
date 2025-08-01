# 🧠 PyTorch Notes

> A beginner-friendly guide to learn PyTorch – the powerful deep learning framework.

---

## 📦 What is PyTorch?

- **PyTorch** is an open-source machine learning library developed by **Facebook AI**.
- It is used for:
  - Deep Learning
  - Computer Vision
  - Natural Language Processing
  - Reinforcement Learning
- Built on **Torch (Lua)**, but uses **Python** and **C++ (backend)**.

---

## 🛠️ Installation

```bash
pip install torch torchvision torchaudio
```

---

## 📚 Core Concepts

### 1. **Tensors**
- The central data structure in PyTorch (similar to NumPy arrays, but with GPU support).
```python
import torch

x = torch.tensor([1.0, 2.0])        # 1D tensor
y = torch.rand(2, 3)                # random 2x3 tensor
z = torch.zeros(3, 3)               # 3x3 tensor with zeros
```

### 2. **Autograd**
- Enables **automatic differentiation**.
- Track operations on tensors for **gradient computation**.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)   # dy/dx = 2x = 4.0
```

### 3. **CUDA (GPU Support)**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor([1.0, 2.0]).to(device)
```

---

## 🔧 Model Building Workflow

### 1. Define the Model
```python
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
```

### 2. Define Loss and Optimizer
```python
model = LinearModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### 3. Training Loop
```python
for epoch in range(100):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 📦 Datasets and DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        self.x = torch.arange(0, 10).float().view(-1, 1)
        self.y = self.x * 2 + 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = MyDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)
```

---

## 🧪 Evaluation

```python
model.eval()
with torch.no_grad():
    predictions = model(x_test)
```

---

## 📘 Common Modules

| Module          | Use                                      |
|-----------------|-------------------------------------------|
| `torch`         | Core tensor operations                    |
| `torch.nn`      | Neural network layers, loss functions     |
| `torch.optim`   | Optimizers (SGD, Adam, etc.)              |
| `torch.utils.data` | Dataset, DataLoader                 |
| `torchvision`   | Image datasets and transformations        |

---

## 🧠 Key Tips

- Use `.to(device)` to leverage GPU
- Use `.detach()` or `with torch.no_grad()` to stop gradient tracking
- Use `model.train()` and `model.eval()` for proper behavior during training/evaluation

---

## 🔗 Resources

- [Official Docs](https://pytorch.org/docs/stable/index.html)
- [60-min Blitz Tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [TorchVision](https://pytorch.org/vision/stable/index.html)
