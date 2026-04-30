# Experiment 4 — Accuracy: SNN vs DNN on MNIST
# Validates Table V: Accuracy (%), showing ~3-4% gap between SNN and DNN
# Run on Google Colab — select Runtime > Change runtime type > GPU for faster training

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
import torchvision
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])

trainset     = torchvision.datasets.MNIST('.', True,  True, transform=transform)
testset      = torchvision.datasets.MNIST('.', False, True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, 128, shuffle=True)
test_loader  = torch.utils.data.DataLoader(testset,  256)

# ── SNN Model ────────────────────────────────────────────────────────────────
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(784, 512)
        self.lif1 = snn.Leaky(beta=0.95, learn_beta=True)
        self.fc2  = nn.Linear(512, 256)
        self.lif2 = snn.Leaky(beta=0.95, learn_beta=True)
        self.fc3  = nn.Linear(256, 10)
        self.lif3 = snn.Leaky(beta=0.95, learn_beta=True)

    def forward(self, x, T=25):
        x  = x.view(x.size(0), -1)
        m1 = self.lif1.init_leaky()
        m2 = self.lif2.init_leaky()
        m3 = self.lif3.init_leaky()
        spk3_all = []
        for _ in range(T):
            s1, m1 = self.lif1(self.fc1(x),  m1)
            s2, m2 = self.lif2(self.fc2(s1), m2)
            s3, m3 = self.lif3(self.fc3(s2), m3)
            spk3_all.append(s3)
        return torch.stack(spk3_all)

# ── DNN Baseline ─────────────────────────────────────────────────────────────
dnn = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 10)
).to(device)

def evaluate_dnn(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs.view(imgs.size(0), -1)).argmax(1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
    return 100 * correct / total

def evaluate_snn(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            spk_out = model(imgs)
            pred    = spk_out.sum(0).argmax(1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
    return 100 * correct / total

# ── Training ─────────────────────────────────────────────────────────────────
EPOCHS    = 5
snn_model = SNN().to(device)
snn_opt   = torch.optim.Adam(snn_model.parameters(), lr=1e-3)
dnn_opt   = torch.optim.Adam(dnn.parameters(), lr=1e-3)
snn_loss_fn = SF.ce_rate_loss()
dnn_loss_fn = nn.CrossEntropyLoss()

snn_accs, dnn_accs = [], []

for epoch in range(EPOCHS):
    # Train SNN
    snn_model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        snn_opt.zero_grad()
        spk_out = snn_model(imgs)
        loss    = snn_loss_fn(spk_out, labels)
        loss.backward()
        snn_opt.step()

    # Train DNN
    dnn.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        dnn_opt.zero_grad()
        dnn_loss_fn(dnn(imgs.view(imgs.size(0), -1)), labels).backward()
        dnn_opt.step()

    snn_acc = evaluate_snn(snn_model, test_loader)
    dnn_acc = evaluate_dnn(dnn, test_loader)
    snn_accs.append(snn_acc)
    dnn_accs.append(dnn_acc)
    print(f"Epoch {epoch+1}/{EPOCHS}  |  SNN: {snn_acc:.2f}%  |  DNN: {dnn_acc:.2f}%")

print(f"\nFinal SNN accuracy : {snn_accs[-1]:.2f}%")
print(f"Final DNN accuracy : {dnn_accs[-1]:.2f}%")
print(f"Accuracy gap       : {dnn_accs[-1] - snn_accs[-1]:.2f}%  (Table V shows ~3.4%)")

# Plot
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS+1), snn_accs, 'o-', label='SNN', color='#2980b9')
plt.plot(range(1, EPOCHS+1), dnn_accs, 's-', label='DNN', color='#c0392b')
plt.xlabel('Epoch')
plt.ylabel('Test accuracy (%)')
plt.title('SNN vs DNN accuracy on MNIST (Table V validation)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('fig_table5_accuracy.png', dpi=150)
plt.show()
print("Saved: fig_table5_accuracy.png")
