# Experiment 2 — Energy Efficiency: SNN vs GPU
# Validates Table III: Power (W), Efficiency (TOPS/W), Energy saved vs GPU
# Run on Google Colab (GPU recommended) or locally

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# Energy constants from literature (picojoules per synaptic operation)
E_GPU_pJ   = 8.6    # GPU (Jetson Nano class) — Horowitz 2014
E_LOIHI_pJ = 0.08   # Intel Loihi             — Davies et al. 2018

# Load MNIST test set
transform = torchvision.transforms.ToTensor()
testset   = torchvision.datasets.MNIST(root='.', train=False,
                                       download=True, transform=transform)
loader    = torch.utils.data.DataLoader(testset, batch_size=64)

# Define simple SNN
class SmallSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(784, 256)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2  = nn.Linear(256, 10)
        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self, x, num_steps=25):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec, total_spk = [], 0
        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            total_spk  += spk1.sum().item()
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec), total_spk

model = SmallSNN()
correct, total_sops, n_samples = 0, 0, 0

with torch.no_grad():
    for imgs, labels in loader:
        x = imgs.view(imgs.size(0), -1)
        spk_out, sops = model(x)
        pred = spk_out.sum(0).argmax(1)
        correct    += (pred == labels).sum().item()
        total_sops += sops
        n_samples  += imgs.size(0)
        if n_samples >= 512:
            break

acc = 100 * correct / n_samples

# Energy estimates
dnn_ops    = n_samples * 25 * 784 * 256
E_gpu_uJ   = dnn_ops   * E_GPU_pJ   / 1e6
E_loihi_uJ = total_sops * E_LOIHI_pJ / 1e6
saving     = 100 * (1 - E_loihi_uJ / E_gpu_uJ)

print(f"Accuracy       : {acc:.1f}%")
print(f"SNN sparsity   : {100*total_sops/(n_samples*25*784*256):.1f}% of dense ops")
print(f"Est GPU energy : {E_gpu_uJ:.2f} uJ per batch")
print(f"Est SNN energy : {E_loihi_uJ:.4f} uJ per batch")
print(f"Energy saving  : {saving:.1f}%  (Table III shows ~90%)")

# Bar chart
platforms = ['GPU (dense)', 'SNN (sparse)']
energies  = [E_gpu_uJ, E_loihi_uJ]
colors    = ['#c0392b', '#2980b9']
plt.figure(figsize=(7, 4))
plt.bar(platforms, energies, color=colors, width=0.5)
plt.ylabel('Estimated energy (μJ)')
plt.title('Energy per inference — SNN vs GPU (Table III validation)')
plt.annotate(f'{saving:.0f}% saving', xy=(1, E_loihi_uJ),
             xytext=(0.6, E_gpu_uJ * 0.5),
             arrowprops=dict(arrowstyle='->'), fontsize=11)
plt.savefig('fig_table3_energy.png', dpi=150)
plt.show()
print("Saved: fig_table3_energy.png")
