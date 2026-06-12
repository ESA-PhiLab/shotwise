# ShotWise

**ShotWise** is a Python toolkit for **task-aware measurement allocation** in noisy kernel estimation.

The package is designed for settings where kernel values are obtained through **stochastic measurements** (e.g., quantum kernels), and measurement resources are limited.

---

## 🚀 First Algorithm: SupportShot

The first algorithm included in ShotWise is:

> **SupportShot** — adaptive shot allocation for kernelized SVMs

SupportShot focuses measurement effort on the **most decision-critical entries** of the kernel matrix by combining:

- **Margin sensitivity** (how kernel entries influence the classifier)
- **Active-set instability** (how likely support vectors are to change)

This leads to:
- better kernel estimates under fixed budgets
- improved classifier accuracy
- reduced measurement cost via early stopping

---

## ✨ Features

- ✅ Adaptive shot allocation for noisy kernels  
- ✅ Designed for kernel SVMs (initial release)  
- ✅ Backend-agnostic architecture  
- ✅ Works with:
  - synthetic simulators (included)
  - Qiskit (planned)
  - PennyLane (planned)
- ✅ Built for noisy, measurement-limited environments  
- ✅ Reproducible experimental pipeline  

---

## 📦 Installation

### Basic installation

```bash
pip install shotwise
``