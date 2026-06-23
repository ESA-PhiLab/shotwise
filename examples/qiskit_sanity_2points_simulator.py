import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from shotwise.backends.qiskit_backend import QiskitBackend

# ============================================================
# 1. Define feature map (1-qubit RY encoding)
# ============================================================

theta = Parameter("θ")

feature_map = QuantumCircuit(1)
feature_map.ry(theta, 0)

print("Feature map:")
print(feature_map.draw())


# ============================================================
# 2. Kernel (SIMULATOR)
# ============================================================

sampler = Sampler()  # local simulator


kernel = FidelityQuantumKernel(
    feature_map=feature_map,
)


# ============================================================
# 3. Data (two points)
# ============================================================

X = np.array(
    [
        [0.0],  # |0>
        [np.pi / 2],  # |+>
    ]
)

print("\nInput points:")
print(X)


# ============================================================
# 4. Backend
# ============================================================

backend = QiskitBackend(
    X,
    kernel=kernel,
    shots=2000,  # large enough to reduce noise
)


# ============================================================
# 5. Shot matrix (only one kernel entry)
# ============================================================

shot_matrix = np.zeros((2, 2), dtype=int)
shot_matrix[0, 1] = 2000


# ============================================================
# 6. Run
# ============================================================

counts = backend.sample_upper_triangle_counts(
    shot_matrix,
    round_id=0,
)

print("\nCounts matrix:")
print(counts)


# ============================================================
# 7. Extract result
# ============================================================

estimated_prob = counts[0, 1] / shot_matrix[0, 1]

print("\nEstimated probability:")
print(estimated_prob)

print("\nExpected probability: 0.5")
