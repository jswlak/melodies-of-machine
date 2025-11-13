#Lesson 2 — Autograd (automatic differentiation)



#Goal: compute gradients automatically for scalar loss.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


x = torch.tensor(2.0, requires_grad=True)
y = x**3 + 2*x      # y = x^3 + 2x
y.backward()        # dy/dx = 3x^2 + 2 evaluated at x=2 -> 3*4 + 2 = 14
print(x.grad)       # prints tensor(14.)
