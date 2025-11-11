#Lesson 0 — Boilerplate (device & imports)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




#output
#--------------

# Using device: cpu

#--------------