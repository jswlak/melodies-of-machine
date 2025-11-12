#Lesson 1 — Tensors (the core)

#Goal: understand creation, shapes, device, basic ops, and conversion to/from NumPy.


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# create tensors
a = torch.tensor([1.0, 2.0, 3.0])        # default float32
b = torch.randn(2, 3)                    # random normal
c = torch.zeros((2,2))

# shapes & device
print(a.shape)      # torch.Size([3])
print(b.size())     # same as shape
print(c.device)

# operations
d = a * 2 + 1
e = a @ torch.tensor([1.0, 1.0, 1.0])    # dot product

# to GPU (if available)
a_gpu = a.to(device)

# convert to numpy
import numpy as np
np_arr = a.numpy()   # only if tensor on CPU
t_from_np = torch.from_numpy(np_arr)
