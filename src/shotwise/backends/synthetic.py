"""Synthetic measurement backend for ShotWise.

This module provides a Bernoulli measurement backend for simulated kernel
estimation experiments. It is intended primarily for testing, development,
and reproducing the paper experiments before integrating hardware-aware
backends such as Qiskit Runtime or PennyLane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from shotwise.utils.matrix import check_square_matrix, upper_triangle_only


@dataclass(frozen=True)
class MeasurementEstimate:
    """Result of estimating a single kernel entry.

    Parameters
    ----------
    i
        Row index of the kernel entry.
    j
        Column index of the kernel entry.
    shots
        Number of measurement shots used.
    successes
        Number of Bernoulli successes observed.
    mean
        Empirical estimate of the kernel entry, equal to successes / shots
        when shots > 0, otherwise 0.0.
    probability_used
        The effective probability used internally for sampling after adding
        synthetic noise and clipping to [0, 1].
    """

    i: int
    j: int
    shots: int
    successes: int
    mean: float
    probability_used: float


class SyntheticBernoulliBackend:
    """Synthetic backend for noisy Bernoulli kernel measurements.

    The backend simulates access to a symmetric kernel matrix whose entries are
    interpreted as Bernoulli success probabilities. This is a natural model for
    overlap-type kernel estimation where each shot produces a binary outcome.

    A measurement call can include two types of synthetic hardware-like noise:

    - global drift: one scalar Gaussian perturbation shared across all queried
      entries in the same call,
    - local fluctuations: independent Gaussian perturbations per queried entry.

    The final effective probability for each entry is clipped to the interval
    [0, 1] before binomial sampling.

    Parameters
    ----------
    kernel_matrix
        Symmetric matrix of Bernoulli probabilities in [0, 1].
    noise_sigma_global
        Standard deviation of the global Gaussian drift term.
    noise_sigma_local
        Standard deviation of the local Gaussian fluctuation term.
    random_state
        Seed or NumPy Generator for reproducible sampling.
    symmetry_tol
        Absolute tolerance used when checking symmetry.

    Raises
    ------
    ValueError
        If the kernel matrix is not square, not symmetric, or contains values
        outside the interval [0, 1].

    Notes
    -----
    The default configuration uses independent Bernoulli sampling without
    additional synthetic drift. Optional global and local Gaussian perturbations
    can be enabled for exploratory experiments.
    """

    def __init__(
        self,
        kernel_matrix: np.ndarray,
        *,
        noise_sigma_global: float = 0.0,
        noise_sigma_local: float = 0.0,
        random_state: int | np.random.Generator | None = None,
        symmetry_tol: float = 1e-10,
    ) -> None:
        check_square_matrix(kernel_matrix, name="kernel_matrix")

        kernel = np.asarray(kernel_matrix, dtype=float)

        if not np.allclose(kernel, kernel.T, atol=symmetry_tol):
            raise ValueError("kernel_matrix must be symmetric.")

        if np.any(kernel < 0.0) or np.any(kernel > 1.0):
            raise ValueError("kernel_matrix entries must lie in the interval [0, 1].")

        if noise_sigma_global < 0.0:
            raise ValueError(
                f"noise_sigma_global must be non-negative, got {noise_sigma_global}."
            )

        if noise_sigma_local < 0.0:
            raise ValueError(
                f"noise_sigma_local must be non-negative, got {noise_sigma_local}."
            )

        self._kernel = kernel.copy()
        self.noise_sigma_global = float(noise_sigma_global)
        self.noise_sigma_local = float(noise_sigma_local)
        self.n = kernel.shape[0]
        self.rng = self._make_rng(random_state)

    @staticmethod
    def _make_rng(
        random_state: int | np.random.Generator | None,
    ) -> np.random.Generator:
        """Create or normalize a NumPy random generator."""
        if isinstance(random_state, np.random.Generator):
            return random_state
        return np.random.default_rng(random_state)

    @property
    def kernel_matrix(self) -> np.ndarray:
        """Return a copy of the underlying true kernel matrix."""
        return self._kernel.copy()

    def estimate_entries(
        self,
        pairs: Iterable[tuple[int, int]],
        shots: int | Sequence[int],
    ) -> list[MeasurementEstimate]:
        """Estimate selected kernel entries using Bernoulli sampling.

        Parameters
        ----------
        pairs
            Iterable of index pairs (i, j). Only off-diagonal entries are
            allowed. Pairs are normalized internally so that i < j.
        shots
            Either a single integer applied to all pairs or a sequence of shot
            counts with the same length as ``pairs``.

        Returns
        -------
        list[MeasurementEstimate]
            One estimate per requested pair.

        Raises
        ------
        ValueError
            If any pair is invalid or if the shot specification is invalid.
        """
        normalized_pairs = [self._normalize_pair(i, j) for i, j in pairs]
        shot_list = self._normalize_shots(shots, len(normalized_pairs))

        if not normalized_pairs:
            return []

        effective_probs = self._sample_effective_probabilities(normalized_pairs)

        estimates: list[MeasurementEstimate] = []
        for (i, j), n_shots, prob in zip(
            normalized_pairs,
            shot_list,
            effective_probs,
            strict=True,
        ):
            successes = int(self.rng.binomial(n_shots, prob)) if n_shots > 0 else 0
            mean = successes / n_shots if n_shots > 0 else 0.0
            estimates.append(
                MeasurementEstimate(
                    i=i,
                    j=j,
                    shots=n_shots,
                    successes=successes,
                    mean=mean,
                    probability_used=float(prob),
                )
            )

        return estimates

    def sample_upper_triangle_counts(self, shot_matrix: np.ndarray) -> np.ndarray:
        """Sample Bernoulli counts for a full shot-allocation matrix.

        Only the strict upper triangle of ``shot_matrix`` is used. The diagonal
        and lower triangle are ignored.

        Parameters
        ----------
        shot_matrix
            Square matrix of requested shot counts.

        Returns
        -------
        np.ndarray
            Integer matrix containing sampled counts in the strict upper
            triangle and zeros elsewhere.
        """
        check_square_matrix(shot_matrix, name="shot_matrix")

        if shot_matrix.shape[0] != self.n:
            raise ValueError(
                "shot_matrix has incompatible shape: "
                f"expected ({self.n}, {self.n}), got {shot_matrix.shape}."
            )

        shots_upper = upper_triangle_only(np.asarray(shot_matrix, dtype=int))

        if np.any(shots_upper < 0):
            raise ValueError("shot_matrix must not contain negative shot counts.")

        rows, cols = np.triu_indices(self.n, k=1)
        pairs = list(zip(rows.tolist(), cols.tolist(), strict=False))
        flat_shots = shots_upper[rows, cols].tolist()

        estimates = self.estimate_entries(pairs, flat_shots)

        counts_upper = np.zeros((self.n, self.n), dtype=int)
        for estimate in estimates:
            counts_upper[estimate.i, estimate.j] = estimate.successes

        return counts_upper

    def _sample_effective_probabilities(
        self,
        pairs: Sequence[tuple[int, int]],
    ) -> np.ndarray:
        """Sample effective probabilities for a batch of queried entries."""
        base_probs = np.array([self._kernel[i, j] for i, j in pairs], dtype=float)

        if self.noise_sigma_global == 0.0 and self.noise_sigma_local == 0.0:
            return base_probs

        global_noise = (
            self.rng.normal(0.0, self.noise_sigma_global)
            if self.noise_sigma_global > 0.0
            else 0.0
        )

        local_noise = (
            self.rng.normal(0.0, self.noise_sigma_local, size=len(pairs))
            if self.noise_sigma_local > 0.0
            else 0.0
        )

        noisy_probs = base_probs + global_noise + local_noise
        return np.clip(noisy_probs, 0.0, 1.0)

    def _normalize_pair(self, i: int, j: int) -> tuple[int, int]:
        """Validate and normalize a matrix index pair."""
        i = int(i)
        j = int(j)

        if i == j:
            raise ValueError("Diagonal entries must not be measured explicitly.")

        if i < 0 or j < 0 or i >= self.n or j >= self.n:
            raise ValueError(
                f"Pair ({i}, {j}) is out of bounds for a {self.n}x{self.n} matrix."
            )

        return (i, j) if i < j else (j, i)

    def _normalize_shots(
        self,
        shots: int | Sequence[int],
        num_pairs: int,
    ) -> list[int]:
        """Normalize a scalar or sequence of shot counts into a list."""
        if isinstance(shots, int):
            if shots < 0:
                raise ValueError(f"shots must be non-negative, got {shots}.")
            return [shots] * num_pairs

        shot_list = [int(s) for s in shots]
        if len(shot_list) != num_pairs:
            raise ValueError(
                "If shots is a sequence, it must have the same length as pairs. "
                f"Got {len(shot_list)} shots for {num_pairs} pairs."
            )

        if any(s < 0 for s in shot_list):
            raise ValueError("Shot counts must be non-negative.")

        return shot_list
