#Lesson 4 — Linear Regression from scratch (end-to-end)

# Train a model y = w*x + b on synthetic data.

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

# 1) Create synthetic data: y = 3*x + 4 + noise
import random
random.seed(0)
X = [[i] for i in range(100)]
y = [3*i + 4 + random.uniform(-5,5) for i in range(100)]

class LinDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

ds = LinDataset(X, y)
loader = DataLoader(ds, batch_size=16, shuffle=True)

# 2) Define model (single linear layer)
model = nn.Linear(1, 1).to(device)

# 3) Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# 4) Training loop
epochs = 300
for epoch in range(epochs):
    epoch_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        preds = model(xb)            # forward
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()              # gradients
        optimizer.step()             # update weights

        epoch_loss += loss.item() * xb.size(0)

    epoch_loss /= len(ds)
    if (epoch+1) % 50 == 0:
        w = model.weight.item()
        b = model.bias.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, w={w:.3f}, b={b:.3f}")

# 5) Inspect learned parameters
print("Learned w, b:", model.weight.item(), model.bias.item())
