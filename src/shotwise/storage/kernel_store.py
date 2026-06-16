"""Kernel estimation storage for ShotWise.

This module provides the stateful storage layer used to accumulate noisy kernel
measurements over multiple rounds of adaptive allocation. It keeps track of the
strict upper-triangular counts and shot budgets, and reconstructs the current
full symmetric kernel estimate on demand.
"""

from __future__ import annotations

import numpy as np

from shotwise.utils.matrix import (
    check_square_matrix,
    reconstruct_symmetric,
    safe_divide,
    upper_triangle_only,
)


class KernelStore:
    """Accumulate and expose a symmetric kernel estimate.

    The store keeps only the independent strict upper-triangular entries of the
    kernel matrix. The diagonal is assumed to be known and is injected only when
    reconstructing the full kernel matrix.

    Parameters
    ----------
    n
        Dimension of the kernel matrix.
    diagonal_value
        Value used to fill the diagonal when reconstructing the full kernel
        matrix. For normalized kernels this is typically 1.0.

    Raises
    ------
    ValueError
        If ``n`` is not positive.
    """

    def __init__(self, n: int, *, diagonal_value: float = 1.0) -> None:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}.")

        self.n = int(n)
        self.diagonal_value = float(diagonal_value)
        self._counts_upper = np.zeros((self.n, self.n), dtype=int)
        self._shots_upper = np.zeros((self.n, self.n), dtype=int)

    @property
    def counts_upper(self) -> np.ndarray:
        """Return a copy of the cumulative upper-triangular success counts."""
        return self._counts_upper.copy()

    @property
    def shots_upper(self) -> np.ndarray:
        """Return a copy of the cumulative upper-triangular shot counts."""
        return self._shots_upper.copy()

    def update(self, counts_upper: np.ndarray, shots_upper: np.ndarray) -> None:
        """Accumulate a new batch of measurements.

        Parameters
        ----------
        counts_upper
            Strict upper-triangular matrix of Bernoulli successes.
        shots_upper
            Strict upper-triangular matrix of allocated shots.

        Raises
        ------
        ValueError
            If the inputs have invalid shape, contain lower-triangular/diagonal
            entries, contain negative values, or if any success count exceeds
            the corresponding number of shots.
        """
        counts = self._validate_upper_integer_matrix(
            counts_upper,
            name="counts_upper",
        )
        shots = self._validate_upper_integer_matrix(
            shots_upper,
            name="shots_upper",
        )

        if np.any(counts > shots):
            raise ValueError("counts_upper must not exceed shots_upper entrywise.")

        self._counts_upper += counts
        self._shots_upper += shots

    def reset(self) -> None:
        """Reset the store to the initial empty state."""
        self._counts_upper.fill(0)
        self._shots_upper.fill(0)

    def kernel_upper(self) -> np.ndarray:
        """Return the current strict upper-triangular kernel estimate."""
        return safe_divide(self._counts_upper, self._shots_upper)

    def kernel_matrix(self) -> np.ndarray:
        """Return the full symmetric kernel estimate."""
        return reconstruct_symmetric(
            self.kernel_upper(),
            diagonal_value=self.diagonal_value,
        )

    def counts_matrix(self) -> np.ndarray:
        """Return a symmetric full matrix of cumulative success counts."""
        return reconstruct_symmetric(self._counts_upper, diagonal_value=0.0)

    def shots_matrix(self) -> np.ndarray:
        """Return a symmetric full matrix of cumulative shot counts."""
        return reconstruct_symmetric(self._shots_upper, diagonal_value=0.0)

    def total_shots_spent(self) -> int:
        """Return the total number of shots accumulated so far."""
        return int(self._shots_upper.sum())

    def num_measured_entries(self) -> int:
        """Return the number of off-diagonal entries measured at least once."""
        return int(np.count_nonzero(self._shots_upper))

    def variance_proxy(self) -> np.ndarray:
        """Return a simple Bernoulli variance proxy for the estimated kernel.

        The returned matrix is the full symmetric matrix with diagonal fixed to
        zero. For entries with zero shots, the variance proxy is set to zero.
        """
        p_upper = self.kernel_upper()
        var_upper = p_upper * (1.0 - p_upper)
        var_upper = safe_divide(var_upper, self._shots_upper)
        return reconstruct_symmetric(var_upper, diagonal_value=0.0)

    def _validate_upper_integer_matrix(
        self,
        matrix: np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        """Validate an update matrix and return a normalized integer copy."""
        array = np.asarray(matrix)
        check_square_matrix(array, name=name)

        if array.shape != (self.n, self.n):
            raise ValueError(
                f"{name} must have shape ({self.n}, {self.n}), got {array.shape}."
            )

        if not np.issubdtype(array.dtype, np.integer):
            if np.issubdtype(array.dtype, np.floating) and np.allclose(
                array,
                np.round(array),
            ):
                array = np.round(array).astype(int)
            else:
                raise ValueError(f"{name} must contain integer values.")
        else:
            array = array.astype(int, copy=True)

        if np.any(array < 0):
            raise ValueError(f"{name} must not contain negative values.")

        expected = upper_triangle_only(array)
        if not np.array_equal(array, expected):
            raise ValueError(
                f"{name} must contain values only in the strict upper triangle."
            )

        return array
