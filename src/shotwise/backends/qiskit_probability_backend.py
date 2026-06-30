"""
Qiskit backend with round-based batched execution.
This backend evaluates kernel probabilities and then locally samples Bernoulli counts!
It's a classical resampling method.

Key features:
- One backend call = one algorithm round
- Circuits grouped into batches/jobs
- Asynchronous job submission with synchronous aggregation
- User-defined quantum kernel (no hardcoded feature map)
- Results mapped back to (i, j) entries
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


class QiskitBackend:
    """
    Quantum backend using batched evaluation per round.

    The backend is kernel-agnostic: the user must provide
    a kernel object with a `.evaluate(X1, X2)` method.
    """

    def __init__(
        self,
        X: np.ndarray,
        *,
        kernel: Any,
        shots: int = 1024,
        max_circuits_per_job: int = 100,
        rng: np.random.Generator | None = None,
    ):
        """
        Parameters
        ----------
        X : np.ndarray
            Input data (n_samples, n_features)

        kernel : object
            Must implement:
                kernel.evaluate(X1, X2) -> ndarray

        shots : int
            Number of shots per evaluation

        max_circuits_per_job : int
            Maximum number of (i, j) evaluations per job batch

        rng : np.random.Generator
            RNG for Bernoulli sampling (optional)
        """
        if not hasattr(kernel, "evaluate"):
            raise ValueError("kernel must implement evaluate(X1, X2)")

        self.X = X
        self.n = X.shape[0]
        self.kernel = kernel
        self.shots = int(shots)
        self.max_circuits_per_job = int(max_circuits_per_job)
        self.rng = rng or np.random.default_rng()

    # ============================================================
    # Main entry point (called once per round)
    # ============================================================

    def sample_upper_triangle_counts(
        self,
        shot_matrix: np.ndarray,
        *,
        round_id: int | None = None,
    ) -> np.ndarray:
        """
        Execute one round of kernel estimation.

        Parameters
        ----------
        shot_matrix : np.ndarray
            Matrix of requested shots per (i, j)

        round_id : int, optional
            Identifier for logging/debugging

        Returns
        -------
        counts : np.ndarray
            Upper-triangular counts matrix
        """
        pairs = self._extract_active_pairs(shot_matrix)

        if not pairs:
            return np.zeros_like(shot_matrix, dtype=int)

        batches = self._split_into_batches(pairs)

        results = self._execute_batches(batches)

        counts = self._results_to_counts(results)

        return counts

    # ============================================================
    # Step 1 — extract (i, j, shots)
    # ============================================================

    def _extract_active_pairs(
        self,
        shot_matrix: np.ndarray,
    ) -> List[Tuple[int, int, int]]:
        rows, cols = np.triu_indices(self.n, k=1)

        pairs = []
        for i, j in zip(rows, cols, strict=True):
            shots = int(shot_matrix[i, j])
            if shots > 0:
                pairs.append((i, j, shots))

        return pairs

    # ============================================================
    # Step 2 — batching
    # ============================================================

    def _split_into_batches(
        self,
        pairs: List[Tuple[int, int, int]],
    ) -> List[List[Tuple[int, int, int]]]:
        batches = []

        for k in range(0, len(pairs), self.max_circuits_per_job):
            batch = pairs[k : k + self.max_circuits_per_job]
            batches.append(batch)

        return batches

    # ============================================================
    # Step 3 — execution (async -> sync)
    # ============================================================

    def _execute_batches(
        self,
        batches: List[List[Tuple[int, int, int]]],
    ) -> List[Tuple[int, int, int, float]]:
        """
        Execute all batches.

        Each batch:
        - evaluated as a matrix call to kernel.evaluate
        - treated as one "job"

        Returns
        -------
        list of (i, j, shots, probability)
        """
        all_results = []

        for batch in batches:
            X1 = np.array([self.X[i] for (i, j, _) in batch])
            X2 = np.array([self.X[j] for (i, j, _) in batch])

            # Evaluate kernel for all pairs in batch
            # Shape: (batch_size,)
            values = self._evaluate_kernel_batch(X1, X2)

            for (i, j, shots), kij in zip(batch, values, strict=True):
                kij = float(kij)
                kij = np.clip(kij, 0.0, 1.0)  # safety

                all_results.append((i, j, shots, kij))

        return all_results

    # ============================================================
    # Kernel evaluation helper
    # ============================================================

    def _evaluate_kernel_batch(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate kernel entries pairwise.

        Guarantees output shape (n_pairs,)
        """
        values = self.kernel.evaluate(X1, X2)

        values = np.asarray(values)

        # Handle (N, N) or (N, 1) or (scalar) cases
        if values.ndim == 2:
            if values.shape[0] == values.shape[1]:
                # diagonal matrix -> take diagonal
                values = np.diag(values)
            elif values.shape[1] == 1:
                values = values[:, 0]
        elif values.ndim == 0:
            values = np.array([values])

        return values

    # ============================================================
    # Step 4 — convert probabilities -> counts
    # ============================================================

    def _results_to_counts(
        self,
        results: List[Tuple[int, int, int, float]],
    ) -> np.ndarray:
        counts = np.zeros((self.n, self.n), dtype=int)

        for i, j, shots, kij in results:
            successes = self.rng.binomial(shots, kij)
            counts[i, j] = successes

        return counts


""" EXAMPLE USAGE
from qiskit.circuit.library import ZZFeatureMapfrom qiskit.c import Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel

feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=2)

kernel = FidelityQuantumKernel(
    feature_map=feature_map,
    sampler=Sampler(),  # or IBM backend sampler
)

backend = QiskitBackend(
    X,
    kernel=kernel,
    shots=1024,
    max_circuits_per_job=50,
)
"""
