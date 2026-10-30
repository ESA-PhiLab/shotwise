"""Spectral diagnostics for positive-semidefiniteness of estimated kernels.

Finite-shot estimation of individual kernel entries does not, in general,
preserve positive semi-definiteness (PSD), even when the underlying exact
kernel matrix is PSD.

This module provides lightweight diagnostics used by the PSD experiments in
the accompanying paper. The diagnostics are independent of the measurement
backend and allocation strategy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _symmetrize(matrix: ArrayLike) -> NDArray[np.float64]:
    """Validate a matrix and return its symmetrized floating-point form."""
    array = np.asarray(matrix, dtype=float)

    if array.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got {array.ndim} dimensions.")

    if array.shape[0] != array.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {array.shape}.")

    if array.shape[0] == 0:
        raise ValueError("Expected a non-empty matrix.")

    if not np.all(np.isfinite(array)):
        raise ValueError("Matrix contains non-finite values.")

    return 0.5 * (array + array.T)


def kernel_eigenvalues(matrix: ArrayLike) -> NDArray[np.float64]:
    """Return eigenvalues of a symmetrized kernel matrix in ascending order."""
    symmetric = _symmetrize(matrix)
    return np.linalg.eigvalsh(symmetric)


def spectral_tolerance(
    matrix: ArrayLike,
    *,
    factor: float = 100.0,
) -> float:
    """Return a scale-aware tolerance for numerical PSD classification.

    The tolerance separates floating-point round-off from materially negative
    eigenvalues. It scales with matrix dimension and spectral magnitude.

    Parameters
    ----------
    matrix
        Square matrix to inspect.
    factor
        Safety multiplier applied to machine precision.

    Returns
    -------
    float
        Numerical tolerance used when classifying eigenvalues as negative.
    """
    symmetric = _symmetrize(matrix)
    values = np.linalg.eigvalsh(symmetric)

    spectral_scale = max(1.0, float(np.max(np.abs(values))))

    return factor * np.finfo(float).eps * symmetric.shape[0] * spectral_scale


def minimum_eigenvalue(matrix: ArrayLike) -> float:
    """Return the smallest eigenvalue of a kernel matrix."""
    values = kernel_eigenvalues(matrix)
    return float(values[0])


def negative_spectral_mass(
    matrix: ArrayLike,
    *,
    tolerance: float | None = None,
) -> float:
    """Return the total magnitude of materially negative eigenvalues.

    The negative spectral mass is

        M_- = sum_{lambda_i < -epsilon} |lambda_i|,

    where epsilon is either supplied explicitly or determined using
    :func:`spectral_tolerance`.
    """
    symmetric = _symmetrize(matrix)
    values = np.linalg.eigvalsh(symmetric)

    eps = spectral_tolerance(symmetric) if tolerance is None else tolerance

    if eps < 0:
        raise ValueError("tolerance must be non-negative.")

    negative = values[values < -eps]
    return float(np.sum(np.abs(negative)))


def relative_negative_spectral_mass(
    matrix: ArrayLike,
    *,
    tolerance: float | None = None,
) -> float:
    """Return negative spectral mass relative to total spectral magnitude.

    The measure is

        rho_- =
            sum_{lambda_i < -epsilon} |lambda_i|
            / sum_i |lambda_i|.

    It is zero when no materially negative eigenvalues are present.
    """
    symmetric = _symmetrize(matrix)
    values = np.linalg.eigvalsh(symmetric)

    eps = spectral_tolerance(symmetric) if tolerance is None else tolerance

    if eps < 0:
        raise ValueError("tolerance must be non-negative.")

    total_mass = float(np.sum(np.abs(values)))

    if total_mass == 0.0:
        return 0.0

    negative_mass = float(np.sum(np.abs(values[values < -eps])))

    return negative_mass / total_mass


def negative_eigenvalue_count(
    matrix: ArrayLike,
    *,
    tolerance: float | None = None,
) -> int:
    """Return the number of materially negative eigenvalues."""
    symmetric = _symmetrize(matrix)
    values = np.linalg.eigvalsh(symmetric)

    eps = spectral_tolerance(symmetric) if tolerance is None else tolerance

    if eps < 0:
        raise ValueError("tolerance must be non-negative.")

    return int(np.count_nonzero(values < -eps))


def is_psd(
    matrix: ArrayLike,
    *,
    tolerance: float | None = None,
) -> bool:
    """Return whether a matrix has no materially negative eigenvalues."""
    symmetric = _symmetrize(matrix)
    values = np.linalg.eigvalsh(symmetric)

    eps = spectral_tolerance(symmetric) if tolerance is None else tolerance

    if eps < 0:
        raise ValueError("tolerance must be non-negative.")

    return bool(values[0] >= -eps)


def psd_diagnostics(
    matrix: ArrayLike,
    *,
    tolerance: float | None = None,
) -> dict[str, float | int | bool]:
    """Return the spectral diagnostics used by the PSD experiments."""
    symmetric = _symmetrize(matrix)
    values = np.linalg.eigvalsh(symmetric)

    eps = spectral_tolerance(symmetric) if tolerance is None else tolerance

    if eps < 0:
        raise ValueError("tolerance must be non-negative.")

    negative = values[values < -eps]

    negative_mass = float(np.sum(np.abs(negative)))
    total_mass = float(np.sum(np.abs(values)))

    relative_mass = negative_mass / total_mass if total_mass > 0.0 else 0.0

    return {
        "lambda_min": float(values[0]),
        "negative_spectral_mass": negative_mass,
        "relative_negative_spectral_mass": relative_mass,
        "negative_eigenvalue_count": int(negative.size),
        "negative_eigenvalue_fraction": float(negative.size / symmetric.shape[0]),
        "is_psd": bool(values[0] >= -eps),
        "spectral_tolerance": float(eps),
    }
