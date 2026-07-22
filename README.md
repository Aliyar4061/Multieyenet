# MultiEyeNet: A Backbone-Diverse Ensemble for Dual-Modality Ocular Disease Classification

**A unified deep learning framework for automated multi-class screening across anterior-segment and fundus imaging, with explicit evaluation of modality-dependent robustness and per-class diagnostic risks.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aliyar4061/Multieyenet/blob/main/colab_setup.ipynb)

---

## 📝 Abstract

**Objectives:** To develop and validate a unified deep learning framework for automated multi-class screening across two anatomically distinct ophthalmic imaging modalities (anterior-segment and fundus) and to evaluate its structural robustness under simulated real-world imaging variability.

**Methods:** Seven backbone architectures, encompassing convolutional (DenseNet-121, EfficientNet-B2, ResNet-50) and transformer-based (ViT-Base, Swin-Tiny) designs, were evaluated. Top-performing configurations were combined into modality-specific, backbone-diverse ensembles optimized via Sequential Least SQuares Programming (SLSQP). Performance was comprehensively assessed via stratified cross-validation, architectural ablation (evaluating Squeeze-and-Excitation and Graph Convolutional modules), and simulated domain-shift perturbations.

**Results:** 
- **Anterior-Segment:** Ensemble achieved 98.85% accuracy (AUC 0.9990). Ablation revealed removing SE/GCN modules improved baseline DenseNet-121 accuracy by 1.13%.
- **Fundus:** Ensemble achieved 96.71% accuracy (AUC 0.9977). Removing SE/GCN modules improved baseline accuracy by a statistically significant 3.66%.
- **Robustness Asymmetry:** Anterior models were highly vulnerable to photometric shifts but resilient to resolution loss, whereas fundus models exhibited the exact inverse susceptibility. Both collapsed under Gaussian noise (~28% accuracy).
- **Clinical Risk:** Granular analysis exposed elevated false-negative rates (FNR) in high-risk conditions: Uveitis (9.22%) and Glaucoma (5.26%).

---

## 🚀 Key Features

- ✅ **Dual-Modality Benchmarking** – Unified evaluation pipeline across 5-class anterior and 4-class fundus datasets.
- ✅ **SLSQP-Optimized Ensemble** – Mathematically constrained weight fusion of top-3 backbones to decorrelate architecture-specific errors.
- ✅ **Controlled Architectural Ablation** – Empirical evidence challenging the universal benefit of SE and GCN modules in specific clinical contexts.
- ✅ **Modality-Dependent Robustness** – Systematic stress-testing under 4 domain shifts (brightness, contrast, Gaussian noise, resolution reduction).
- ✅ **Per-Class Safety Analysis** – Explicit reporting of False Negative Rates (FNR) and bootstrapped 95% Confidence Intervals for clinical transparency.
- ✅ **XAI Integration** – Grad-CAM visualizations for both CNN and ViT backbones to ensure anatomical plausibility.

---

## 📊 Dataset Summary

| Modality | Total Samples | Classes / Conditions |
|----------|---------------|----------------------|
| **Anterior-Segment** | 2,298 | 5 (Normal, Cataract, Conjunctivitis, Uveitis, Eyelid Drooping) |
| **Fundus** | 4,217 | 4 (Normal, Cataract, Diabetic Retinopathy, Glaucoma) |

*Note: Datasets were stratified into 70% training, 15% validation, and 15% testing splits.*

---

## 📈 Performance Highlights

| Metric | Anterior-Segment Ensemble | Fundus Ensemble |
|--------|---------------------------|-----------------|
| **Accuracy** | 98.85% | 96.71% |
| **AUC (Macro)** | 0.9990 | 0.9977 |
| **Cohen's κ** | 0.9851 | 0.9560 |
| **Critical FNR** | 9.22% (Uveitis) | 5.26% (Glaucoma) |

---

## 🛠 Implementation Details

All experiments are implemented in **PyTorch 2.0+** with the `timm` library for pretrained backbone initialization. 

### Training Hyperparameters & Software Stack
| Component | Specification |
|-----------|---------------|
| **Framework** | PyTorch 2.0+, torchvision 0.15+ |
| **Backbones** | DenseNet-121, EfficientNet-B2, ResNet-50, ViT-Base, Swin-Tiny |
| **Input Size** | 224 × 224 pixels |
| **Batch Size** | 64 (optimized for hybrid training) |
| **Optimizer** | AdamW (lr = 5e-4, weight decay = 1e-2) |
| **Scheduler** | Linear Warm-up (3 epochs) + Cosine Annealing |
| **Loss Function** | Focal Loss (γ = 2.0) + Label Smoothing (ε = 0.1) |
| **Max Epochs** | 30 (Early stopping patience = 3) |
| **Data Augmentation** | Flip (p=0.5), Rotation ±20° (p=0.7), Color Jitter ±0.1, Coarse Dropout, CLAHE, Elastic Transform |
| **Test-Time Augmentation (TTA)** | 5 steps (flips, rotation ±15°) |
| **Ensemble Weighting** | SLSQP optimization (minimizing Log-Loss on validation set) |
| **Statistical Rigor** | 1000-iteration bootstrapping for 95% CIs; Wilcoxon & Kruskal-Wallis tests |

---

## 💻 Environment & Hardware

### System Requirements
- **OS:** Linux (Ubuntu 20.04+), Windows 10/11, or macOS (CPU-only mode)  
- **Python:** 3.8 – 3.10  
- **CUDA:** 11.7 or 11.8 (required for GPU training)  
- **GPU:** NVIDIA RTX 3060 (12 GB VRAM) minimum; RTX 3090 / A100 recommended  
- **RAM:** 32 GB or higher  

### Software Dependencies
All required packages are listed in `requirements.txt`. Core libraries include:
- `torch>=2.0.0`, `torchvision>=0.15.0`
- `timm>=0.9.0` (ImageNet pretrained backbones)
- `albumentations>=1.3.0` (advanced augmentations)
- `scikit-learn`, `pandas`, `numpy`, `scipy` (for SLSQP and statistical tests)
- `matplotlib`, `seaborn`, `tqdm`
- `opencv-python-headless`, `pillow`

---

## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Aliyar4061/Multieyenet.git
cd Multieyenet
pip install -r requirements.txt



