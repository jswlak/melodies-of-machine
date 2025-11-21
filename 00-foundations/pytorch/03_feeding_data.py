#Lesson 3 — Dataset & DataLoader (feeding data)

# Goal: learn how to wrap data for batching and shuffling.

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# example
X = [[i] for i in range(100)]
y = [2*i + 1 for i in range(100)]
ds = SimpleDataset(X, y)
loader = DataLoader(ds, batch_size=16, shuffle=True)
for xb, yb in loader:
    print(xb.shape, yb.shape)
    break

    
