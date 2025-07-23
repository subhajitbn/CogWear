
# model_utils.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----- Common preprocessing -----
def create_windows(signal, window_size=256, step_size=64):
    windows = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start+window_size]
        windows.append(window)
    return np.stack(windows)

def normalize_windows(windows):
    return (windows - windows.mean(axis=1, keepdims=True)) / (windows.std(axis=1, keepdims=True) + 1e-6)

class BVPWindowDataset(Dataset):
    def __init__(self, signal, target, window_size=256, step_size=64):
        raw_windows = create_windows(signal, window_size, step_size)
        norm_windows = normalize_windows(raw_windows)
        self.X = torch.tensor(norm_windows[:, None, :], dtype=torch.float32)
        self.y = torch.full((len(self.X),), target, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----- Common model -----
class EncoderCNN(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

# ----- Shared training loop -----
def train_model(signal, target, window_size=256, step_size=64, epochs=10, lr=1e-3):
    dataset = BVPWindowDataset(signal, target, window_size, step_size)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = EncoderCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in loader:
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    return model, dataset

# ----- Plotting -----
def plot_cl(model, dataset, title="Estimated CL(t)"):
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _ in DataLoader(dataset, batch_size=64):
            preds.extend(model(X).cpu().numpy())

    plt.figure(figsize=(14, 4))
    plt.plot(preds, label="CL(t)")
    plt.title(title)
    plt.xlabel("Window Index")
    plt.ylabel("Cognitive Load")
    plt.grid(True)
    plt.legend()
    plt.show()

    return np.array(preds)
