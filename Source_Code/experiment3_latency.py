# Experiment 3 — Inference Latency: CNN vs SNN
# Validates Table IV: Latency (ms) and improvement vs CNN baseline
# Run on Google Colab or locally

import torch
import torch.nn as nn
import snntorch as snn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time

# ── Models ───────────────────────────────────────────────────────────────────

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10))
    def forward(self, x):
        return self.net(x)

class SmallSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(784, 256)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2  = nn.Linear(256, 10)
        self.lif2 = snn.Leaky(beta=0.9)
    def forward(self, x, T=20):
        m1 = self.lif1.init_leaky()
        m2 = self.lif2.init_leaky()
        out = []
        for _ in range(T):
            s1, m1 = self.lif1(self.fc1(x), m1)
            s2, m2 = self.lif2(self.fc2(s1), m2)
            out.append(s2)
        return torch.stack(out)

cnn       = SmallCNN().eval()
snn_model = SmallSNN().eval()

# Load MNIST
transform = torchvision.transforms.ToTensor()
dataset   = torchvision.datasets.MNIST('.', False, download=True, transform=transform)

batch_sizes = [1, 4, 8, 16, 32]
cnn_lat, snn_lat = [], []

with torch.no_grad():
    for bs in batch_sizes:
        imgs = torch.stack([dataset[i][0] for i in range(bs)]).view(bs, -1)

        reps = 50
        t0 = time.time()
        for _ in range(reps):
            cnn(imgs)
        cnn_lat.append(1000 * (time.time() - t0) / reps)

        t0 = time.time()
        for _ in range(reps):
            snn_model(imgs)
        snn_lat.append(1000 * (time.time() - t0) / reps)

print("Batch | CNN (ms) | SNN (ms) | Improvement")
print("-" * 45)
for i, bs in enumerate(batch_sizes):
    imp = 100 * (cnn_lat[i] - snn_lat[i]) / cnn_lat[i]
    print(f"{bs:>5} | {cnn_lat[i]:>8.2f} | {snn_lat[i]:>8.2f} | {imp:>+.1f}%")

plt.figure(figsize=(8, 4))
plt.plot(batch_sizes, cnn_lat, 'o-', label='CNN (frame-based)', color='#c0392b')
plt.plot(batch_sizes, snn_lat, 's-', label='SNN (event-driven)', color='#2980b9')
plt.xlabel('Batch size')
plt.ylabel('Latency (ms)')
plt.title('Inference latency — CNN vs SNN (Table IV validation)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('fig_table4_latency.png', dpi=150)
plt.show()
print("Saved: fig_table4_latency.png")
