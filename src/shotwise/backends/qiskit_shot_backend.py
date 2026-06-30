"""Qiskit shot backend for direct adaptive quantum-kernel measurement.

This backend differs from the probability-resampling QiskitBackend.

Instead of:
    kernel.evaluate(...) -> probability -> local binomial sampling

it performs:
    shot_matrix -> Qiskit Sampler shots -> actual all-zero counts

This is the backend to use for FakeFez / Aer noisy simulations and real IBM
hardware when the adaptive allocation should control the actual number of
quantum shots.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.passmanager import BasePassManager


@dataclass(frozen=True)
class QiskitShotJobRecord:
    """Metadata for one submitted sampler job."""

    round_id: int | None
    n_circuits: int
    total_shots: int
    job_id: str | None


class QiskitShotBackend:
    """Direct-shot Qiskit backend for adaptive kernel measurement.

    Parameters
    ----------
    X
        Input data with shape ``(n_samples, n_features)``.
    feature_map
        Parameterized quantum feature map ``U(x)``.
    sampler
        Qiskit SamplerV2-compatible primitive. For IBM hardware this is usually
        ``qiskit_ibm_runtime.SamplerV2(mode=backend)``. For local noisy
        simulation this may be ``SamplerV2(mode=AerSimulator.from_backend(...))``.
    pass_manager
        Optional pass manager used to transpile the compute-uncompute circuits.
        Required for IBM Runtime / fake-backend ISA compatibility.
    input_parameters
        Optional explicit ordering of feature-map parameters. If ``None``, the
        parameters are taken from ``feature_map.parameters``.
    max_circuits_per_job
        Maximum circuits submitted in one sampler call.
    use_pub_shots
        If True, submit one sampler job per batch with per-circuit shots encoded
        as PUB-specific shot values. If False, group circuits by identical shot
        count and call ``sampler.run(circuits, shots=...)``.
    """

    def __init__(
        self,
        X: np.ndarray,
        *,
        feature_map: QuantumCircuit,
        sampler: Any,
        pass_manager: BasePassManager | None = None,
        input_parameters: Sequence[Any] | None = None,
        max_circuits_per_job: int = 100,
        use_pub_shots: bool = True,
    ) -> None:
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {self.X.shape}.")

        self.n = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.feature_map = feature_map
        self.sampler = sampler
        self.pass_manager = pass_manager
        self.max_circuits_per_job = int(max_circuits_per_job)
        self.use_pub_shots = bool(use_pub_shots)

        if self.max_circuits_per_job <= 0:
            raise ValueError(
                "max_circuits_per_job must be strictly positive, got "
                f"{self.max_circuits_per_job}."
            )

        if input_parameters is None:
            self.input_parameters = list(feature_map.parameters)
        else:
            self.input_parameters = list(input_parameters)

        if len(self.input_parameters) != self.n_features:
            raise ValueError(
                "Number of input parameters must match X.shape[1]. "
                f"Got {len(self.input_parameters)} parameters and "
                f"{self.n_features} features."
            )

        if feature_map.num_qubits <= 0:
            raise ValueError("feature_map must contain at least one qubit.")

        self.num_qubits = feature_map.num_qubits
        self.job_history_: list[QiskitShotJobRecord] = []
        self._circuit_cache: dict[tuple[int, int], QuantumCircuit] = {}

    # ------------------------------------------------------------------
    # Public API expected by SupportShot
    # ------------------------------------------------------------------

    def sample_upper_triangle_counts(
        self,
        shot_matrix: np.ndarray,
        *,
        round_id: int | None = None,
    ) -> np.ndarray:
        """Measure requested upper-triangular entries.

        Parameters
        ----------
        shot_matrix
            Matrix of requested shots per pair. Only strict upper-triangular
            entries are used.
        round_id
            Optional round identifier for job metadata.

        Returns
        -------
        np.ndarray
            Strict upper-triangular matrix of measured success counts.
        """
        shot_matrix = np.asarray(shot_matrix, dtype=int)
        if shot_matrix.shape != (self.n, self.n):
            raise ValueError(
                f"shot_matrix must have shape {(self.n, self.n)}, "
                f"got {shot_matrix.shape}."
            )

        pairs = self._extract_active_pairs(shot_matrix)
        counts = np.zeros((self.n, self.n), dtype=int)

        if not pairs:
            return counts

        batches = self._split_into_batches(pairs)

        for batch in batches:
            batch_counts = self._run_batch(batch, round_id=round_id)
            for i, j, success_count in batch_counts:
                counts[i, j] = success_count

        return counts

    # ------------------------------------------------------------------
    # Pair extraction and batching
    # ------------------------------------------------------------------

    def _extract_active_pairs(
        self,
        shot_matrix: np.ndarray,
    ) -> list[tuple[int, int, int]]:
        rows, cols = np.triu_indices(self.n, k=1)

        pairs: list[tuple[int, int, int]] = []
        for i, j in zip(rows, cols, strict=True):
            shots = int(shot_matrix[i, j])
            if shots > 0:
                pairs.append((int(i), int(j), shots))

        return pairs

    def _split_into_batches(
        self,
        pairs: list[tuple[int, int, int]],
    ) -> list[list[tuple[int, int, int]]]:
        return [
            pairs[start : start + self.max_circuits_per_job]
            for start in range(0, len(pairs), self.max_circuits_per_job)
        ]

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def _bound_feature_map(self, x: np.ndarray) -> QuantumCircuit:
        values = np.asarray(x, dtype=float)
        if values.shape != (self.n_features,):
            raise ValueError(
                f"Feature vector must have shape ({self.n_features},), "
                f"got {values.shape}."
            )

        parameter_values = dict(zip(self.input_parameters, values, strict=True))
        return self.feature_map.assign_parameters(parameter_values, inplace=False)

    def _build_fidelity_circuit(self, i: int, j: int) -> QuantumCircuit:
        """Build measured compute-uncompute circuit for pair (i, j).

        The circuit prepares ``U(x_i)|0>`` and then applies ``U(x_j)^dagger``.
        The probability of measuring all zeros is the fidelity kernel entry.
        """
        cache_key = (i, j)
        cached = self._circuit_cache.get(cache_key)
        if cached is not None:
            return cached

        circuit_i = self._bound_feature_map(self.X[i])
        circuit_j = self._bound_feature_map(self.X[j])

        qc = QuantumCircuit(self.num_qubits)
        qc.compose(circuit_i, inplace=True)
        qc.compose(circuit_j.inverse(), inplace=True)
        qc.measure_all()

        if self.pass_manager is not None:
            qc = self.pass_manager.run(qc)

        self._circuit_cache[cache_key] = qc
        return qc

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _run_batch(
        self,
        batch: list[tuple[int, int, int]],
        *,
        round_id: int | None,
    ) -> list[tuple[int, int, int]]:
        if self.use_pub_shots:
            return self._run_batch_with_pub_shots(batch, round_id=round_id)

        return self._run_batch_grouped_by_shots(batch, round_id=round_id)

    def _run_batch_with_pub_shots(
        self,
        batch: list[tuple[int, int, int]],
        *,
        round_id: int | None,
    ) -> list[tuple[int, int, int]]:
        """Run one sampler job with PUB-specific shot counts.

        This is preferable because it reduces job fragmentation when different
        pairs have different requested shots.
        """
        pubs = []
        meta: list[tuple[int, int, int]] = []

        for i, j, shots in batch:
            circuit = self._build_fidelity_circuit(i, j)
            pubs.append((circuit, {}, shots))
            meta.append((i, j, shots))

        job = self.sampler.run(pubs)
        result = job.result()

        job_id = self._safe_job_id(job)
        self.job_history_.append(
            QiskitShotJobRecord(
                round_id=round_id,
                n_circuits=len(pubs),
                total_shots=sum(shots for _, _, shots in meta),
                job_id=job_id,
            )
        )

        return self._extract_counts_from_result(result, meta)

    def _run_batch_grouped_by_shots(
        self,
        batch: list[tuple[int, int, int]],
        *,
        round_id: int | None,
    ) -> list[tuple[int, int, int]]:
        """Fallback path: one sampler job per distinct shot count."""
        output: list[tuple[int, int, int]] = []

        shots_to_pairs: dict[int, list[tuple[int, int, int]]] = {}
        for i, j, shots in batch:
            shots_to_pairs.setdefault(shots, []).append((i, j, shots))

        for shots, group in shots_to_pairs.items():
            circuits = [self._build_fidelity_circuit(i, j) for i, j, _ in group]

            job = self.sampler.run(circuits, shots=shots)
            result = job.result()

            job_id = self._safe_job_id(job)
            self.job_history_.append(
                QiskitShotJobRecord(
                    round_id=round_id,
                    n_circuits=len(circuits),
                    total_shots=shots * len(circuits),
                    job_id=job_id,
                )
            )

            output.extend(self._extract_counts_from_result(result, group))

        return output

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    def _extract_counts_from_result(
        self,
        result: Any,
        meta: list[tuple[int, int, int]],
    ) -> list[tuple[int, int, int]]:
        if len(result) != len(meta):
            raise RuntimeError(
                "Sampler result length does not match submitted circuit count. "
                f"Got {len(result)} results and {len(meta)} metadata entries."
            )

        output: list[tuple[int, int, int]] = []

        for pub_result, (i, j, shots) in zip(result, meta, strict=True):
            counts = self._get_counts(pub_result)
            success_key = "0" * self._num_measured_bits(pub_result)
            successes = int(counts.get(success_key, 0))

            if successes < 0 or successes > shots:
                raise RuntimeError(
                    f"Invalid success count {successes} for shots={shots} "
                    f"on pair ({i}, {j})."
                )

            output.append((i, j, successes))

        return output

    def _get_counts(self, pub_result: Any) -> dict[str, int]:
        """Return combined counts from a SamplerV2 PUB result."""
        if hasattr(pub_result, "join_data"):
            return pub_result.join_data().get_counts()

        # Fallback for simple measure_all() circuits where the register is
        # usually named "meas".
        if hasattr(pub_result.data, "meas"):
            return pub_result.data.meas.get_counts()

        # Last-resort fallback: use the first BitArray-like register.
        for name in pub_result.data:
            return pub_result.data[name].get_counts()

        raise RuntimeError("Could not extract counts from sampler result.")

    def _num_measured_bits(self, pub_result: Any) -> int:
        """Infer number of measured classical bits from the joined data."""
        if hasattr(pub_result, "join_data"):
            joined = pub_result.join_data()
            if hasattr(joined, "num_bits"):
                return int(joined.num_bits)

        # For our circuits, all feature-map qubits are measured.
        return self.num_qubits

    def _safe_job_id(self, job: Any) -> str | None:
        if hasattr(job, "job_id"):
            job_id = job.job_id
            if callable(job_id):
                return str(job_id())
            return str(job_id)
        return None
