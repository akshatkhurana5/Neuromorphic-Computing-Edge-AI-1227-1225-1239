# Neuromorphic Computing Chips for Low-Power Edge AI Applications

---

## Team Details

| Roll Number | Name |
|-------------|------|
| BE22-CSE-2210991227 | Akshat Khurana |
| BE22-CSE-2210991225 | Akshat Gupta |
| BE22-CSE-2210991239 | Akshit Mishra |

**Department:** Dept. of Computer Science Engineering  
**University:** Chitkara University, Punjab, India  
**Project Type:** Research Paper (IPR Submission)  
**Current Status:** Submitted for Publication

---

## Project Title
**Neuromorphic Computing Chips for Low-Power Edge AI Applications**

---

## Abstract
Modern artificial intelligence (AI) systems demand increasing computational power, yet edge devices operate under strict energy constraints. This research investigates neuromorphic computing chips — hardware that mimics the brain's spiking neural network (SNN) architecture — as a solution for deploying AI at the edge with ultra-low power consumption.

---

## Repository Structure

```
Neuromorphic-Computing-Edge-AI/
│
├── IPR_Submission_Proof/
│   └── [Screenshot of submission / copyright form]
│
├── Report_and_PPT/
│   └── [Research paper .docx and presentation .pptx]
│
├── Source_Code/
│   ├── experiment1_lif_scaling.py       # Table II — LIF neuron scaling (Brian2)
│   ├── experiment2_energy_efficiency.py # Table III — SNN vs GPU energy (SNNTorch)
│   ├── experiment3_latency.py           # Table IV — Inference latency comparison
│   ├── experiment4_accuracy.py          # Table V — SNN vs DNN accuracy (MNIST)
│   └── requirements.txt                 # All dependencies
│
└── README.md
```

---

## Technologies Used

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.x | Base language |
| Brian2 | 2.x | LIF neuron simulation |
| SNNTorch | Latest | SNN training & inference |
| PyTorch | Latest | Deep learning framework |
| TorchVision | Latest | MNIST dataset loading |
| NumPy | Latest | Numerical computation |
| Matplotlib | Latest | Figure generation |
| Google Colab | — | Execution environment (free GPU) |

---

## How to Run

### 1. Install dependencies
```bash
pip install brian2 snntorch torch torchvision numpy matplotlib
```

### 2. Run experiments
```bash
# Experiment 1 — LIF Neuron Scaling (validates Table II)
python Source_Code/experiment1_lif_scaling.py

# Experiment 2 — Energy Efficiency (validates Table III)
python Source_Code/experiment2_energy_efficiency.py

# Experiment 3 — Inference Latency (validates Table IV)
python Source_Code/experiment3_latency.py

# Experiment 4 — SNN vs DNN Accuracy (validates Table V)
python Source_Code/experiment4_accuracy.py
```

### 3. Or run on Google Colab
Open each `.py` file in Google Colab by uploading it, or paste the code directly into a new notebook cell.

---

## Collaborator
This repository is shared with: **cse.ph4e@chitkara.edu.in**
