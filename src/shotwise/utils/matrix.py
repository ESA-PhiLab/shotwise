"""Matrix utilities for ShotWise.

This module centralizes low-level helpers for working with symmetric kernel
matrices and their independent upper-triangular entries.
"""

from __future__ import annotations

import numpy as np


def check_square_matrix(matrix: np.ndarray, *, name: str = "matrix") -> None:
    """Validate that an array is a 2D square matrix.

    Parameters
    ----------
    matrix
        Input array to validate.
    name
        Human-readable name used in error messages.

    Raises
    ------
    ValueError
        If the input is not a 2D square array.
    """
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got ndim={matrix.ndim}.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got shape={matrix.shape}.")


def upper_triangle_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return indices for the strict upper triangle of an n x n matrix.

    Parameters
    ----------
    n
        Matrix dimension.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Row and column index arrays for entries (i, j) with i < j.

    Raises
    ------
    ValueError
        If n is negative.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}.")
    return np.triu_indices(n, k=1)


def num_upper_triangle_entries(n: int) -> int:
    """Return the number of strict upper-triangle entries in an n x n matrix."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}.")
    return n * (n - 1) // 2


def upper_triangle_only(matrix: np.ndarray) -> np.ndarray:
    """Return a copy containing only the strict upper triangle of a matrix.

    The diagonal and lower-triangular entries are set to zero.

    Parameters
    ----------
    matrix
        Input square matrix.

    Returns
    -------
    np.ndarray
        Matrix with only the strict upper triangle retained.
    """
    check_square_matrix(matrix)
    return np.triu(matrix, k=1)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Safely divide two arrays elementwise.

    Returns zero wherever the denominator is zero.

    Parameters
    ----------
    numerator
        Numerator array.
    denominator
        Denominator array.

    Returns
    -------
    np.ndarray
        Elementwise quotient with zeros where denominator == 0.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    if numerator.shape != denominator.shape:
        raise ValueError(
            "numerator and denominator must have the same shape, "
            f"got {numerator.shape} and {denominator.shape}."
        )

    out = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def reconstruct_symmetric(
    upper: np.ndarray,
    *,
    diagonal_value: float = 0.0,
) -> np.ndarray:
    """Reconstruct a full symmetric matrix from its strict upper triangle.

    Parameters
    ----------
    upper
        Matrix whose strict upper triangle contains the independent entries.
        Lower-triangular entries are ignored.
    diagonal_value
        Value used to fill the diagonal.

    Returns
    -------
    np.ndarray
        Full symmetric matrix.
    """
    check_square_matrix(upper, name="upper")
    upper_only = upper_triangle_only(upper)
    full = upper_only + upper_only.T
    np.fill_diagonal(full, diagonal_value)
    return full
