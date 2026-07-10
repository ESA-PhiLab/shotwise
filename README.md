# ShotWise

**ShotWise** is a Python toolkit for adaptive measurement allocation in noisy kernel learning.

The package targets settings where kernel entries are obtained through stochastic measurements and measurement resources are limited. Rather than allocating measurements uniformly across the kernel matrix, ShotWise allocates resources where they matter most for downstream learning tasks.

Typical applications include:

- Quantum kernel methods
- Measurement-limited kernel learning
- Kernelized SVMs
- Gaussian Processes
- Other kernel methods trained from noisy kernel estimates

---

## Motivation

Many kernel-learning pipelines implicitly assume exact access to the kernel matrix.

In emerging settings such as quantum machine learning, this assumption no longer holds. Kernel entries must be estimated from finite numbers of measurements, introducing uncertainty that depends directly on how measurement resources are allocated.

ShotWise provides algorithms that transform measurement allocation from a uniform estimation problem into a task-aware learning problem.

---

## First Algorithm: SupportShot

The first algorithm implemented in ShotWise is:

> **SupportShot** — adaptive measurement allocation for kernelized Support Vector Machines.

SupportShot combines two complementary signals:

- **Margin sensitivity**
  - How strongly a kernel entry influences the SVM solution.

- **Active-set instability**
  - How likely support-vector membership is to change under kernel uncertainty.

These signals define an adaptive allocation policy that concentrates measurements in classifier-relevant regions of the kernel matrix.

Under fixed budgets, this can improve:

- support-vector recovery,
- margin estimation,
- decision-function reconstruction,
- overall classifier fidelity.

---

## Features

- Adaptive allocation for noisy kernel estimation
- SupportShot algorithm for kernelized SVMs
- Backend-agnostic architecture
- Synthetic Bernoulli measurement backends
- Qiskit simulator support
- Qiskit hardware support
- Support for adaptive, uniform, and custom allocation policies
- Reproducible experimental and benchmarking pipeline
- Statistical analysis utilities for large experiment campaigns

---

## Installation

### Basic installation

```bash
pip install shotwise
```

### Development installation

```bash
git clone <repository-url>
cd shotwise

pip install -e .
```

### Optional Qiskit support

```bash
pip install -e .[qiskit]
```

### Qiskit + analysis tools

```bash
pip install -e .[qiskit,examples]
```

---

## Authentication

ShotWise does not manage IBM Quantum authentication.

Configure your IBM Quantum account using:

```bash
qiskit_ibm_runtime save-account --token YOUR_TOKEN
```

or environment variables supported by Qiskit Runtime.

**Never store authentication tokens in source code.**

---

## Quick Start

Synthetic examples:

```bash
python examples/synthetic_backend.py
```

```bash
python examples/blob_benchmark.py
```

These examples run the adaptive allocation pipeline using synthetic noisy kernel measurements.

---

## Hardware Experiments

Example scripts for quantum kernel experiments are available in:

```text
examples/qiskit_toy8_simulator.py
examples/qiskit_toy8_Manila.py
examples/qiskit_toy8_ibm_shot.py
```

Utilities for reproducing the experimental analysis used in the accompanying papers:

```text
examples/analyze_toy_experiments.py
examples/make_ibm_main_tables.py
```

---

## Testing

Run the complete test suite:

```bash
pytest -q
```

Run linting:

```bash
ruff check .
```

---

## Research Status

ShotWise is an active research project.

Current methods focus on adaptive measurement allocation for kernelized SVMs. Ongoing work includes:

- Gaussian Processes
- Additional kernel-learning methods
- New allocation policies
- Hardware-aware adaptive strategies
- Resource-aware kernel learning beyond SVMs

---

## Citation

If you use ShotWise in academic work, please cite the accompanying publication.

Citation information will be added upon publication.

---

## License

License information will be added before the first public release.