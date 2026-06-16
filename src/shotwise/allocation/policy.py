"""Shot-allocation policies for ShotWise.

This module converts upper-triangular score matrices into integer-valued
shot-allocation matrices under a fixed budget. The initial implementation is
intentionally simple and mirrors the multinomial allocation strategy used in
the original proof-of-concept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shotwise.utils.matrix import (
    check_square_matrix,
    upper_triangle_indices,
    upper_triangle_only,
)


@dataclass(frozen=True)
class AllocationResult:
    """Result of converting scores into an integer shot allocation.

    Parameters
    ----------
    budget
        Total number of shots requested.
    allocation_matrix
        Strict upper-triangular integer matrix of allocated shots.
    probabilities
        Flat probability vector over the independent upper-triangular
        entries, ordered according to ``numpy.triu_indices(n, k=1)``.
    scores_upper
        Flat vector of the non-negative upper-triangular scores used to
        construct ``probabilities``.
    """

    budget: int
    allocation_matrix: np.ndarray
    probabilities: np.ndarray
    scores_upper: np.ndarray


def normalize_score_matrix(scores: np.ndarray) -> np.ndarray:
    """Validate and normalize a score matrix to strict upper-triangular form.

    Parameters
    ----------
    scores
        Candidate score matrix.

    Returns
    -------
    np.ndarray
        Strict upper-triangular floating-point score matrix.

    Raises
    ------
    ValueError
        If the matrix is not square or contains negative values.
    """
    score_matrix = np.asarray(scores, dtype=float)
    check_square_matrix(score_matrix, name="scores")

    if np.any(score_matrix < 0.0):
        raise ValueError("scores must not contain negative values.")

    return upper_triangle_only(score_matrix)


def upper_triangle_probability_vector(
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a score matrix into a probability vector over i < j entries.

    Parameters
    ----------
    scores
        Score matrix to normalize.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(scores_upper, probabilities)`` where both are flat arrays ordered
        according to ``numpy.triu_indices(n, k=1)``.

    Raises
    ------
    ValueError
        If the total score is zero.
    """
    score_matrix = normalize_score_matrix(scores)
    rows, cols = upper_triangle_indices(score_matrix.shape[0])
    scores_upper = score_matrix[rows, cols]
    total = float(scores_upper.sum())

    if total <= 0.0:
        raise ValueError("scores must have strictly positive total mass.")

    probabilities = scores_upper / total
    return scores_upper, probabilities


def allocate_multinomial(
    scores: np.ndarray,
    budget: int,
    *,
    random_state: int | np.random.Generator | None = None,
) -> AllocationResult:
    """Allocate a shot budget according to a multinomial draw.

    Parameters
    ----------
    scores
        Non-negative score matrix. Only the strict upper triangle is used.
    budget
        Total number of shots to allocate.
    random_state
        Optional NumPy seed or generator for reproducibility.

    Returns
    -------
    AllocationResult
        Integer-valued upper-triangular allocation and associated
        probabilities.

    Raises
    ------
    ValueError
        If ``budget`` is negative or if the total score is zero when
        ``budget > 0``.
    """
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}.")

    score_matrix = normalize_score_matrix(scores)
    n = score_matrix.shape[0]
    rows, cols = upper_triangle_indices(n)

    if budget == 0:
        zeros = np.zeros_like(rows, dtype=float)
        return AllocationResult(
            budget=0,
            allocation_matrix=np.zeros((n, n), dtype=int),
            probabilities=zeros,
            scores_upper=zeros,
        )

    scores_upper, probabilities = upper_triangle_probability_vector(score_matrix)
    rng = _make_rng(random_state)
    draws = rng.multinomial(int(budget), probabilities)

    allocation = np.zeros((n, n), dtype=int)
    allocation[rows, cols] = draws

    return AllocationResult(
        budget=int(budget),
        allocation_matrix=allocation,
        probabilities=probabilities,
        scores_upper=scores_upper,
    )


def allocate_deterministic_rounding(
    scores: np.ndarray,
    budget: int,
) -> AllocationResult:
    """Allocate a shot budget using deterministic floor-plus-remainder rounding.

    This is a useful alternative to multinomial sampling for tests or for
    deterministic experiment execution.
    """
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}.")

    score_matrix = normalize_score_matrix(scores)
    n = score_matrix.shape[0]
    rows, cols = upper_triangle_indices(n)

    if budget == 0:
        zeros = np.zeros_like(rows, dtype=float)
        return AllocationResult(
            budget=0,
            allocation_matrix=np.zeros((n, n), dtype=int),
            probabilities=zeros,
            scores_upper=zeros,
        )

    scores_upper, probabilities = upper_triangle_probability_vector(score_matrix)
    fractional = budget * probabilities
    base = np.floor(fractional).astype(int)
    remainder = int(budget - base.sum())

    if remainder > 0:
        fractional_parts = fractional - base
        order = np.argsort(fractional_parts)[::-1]
        base[order[:remainder]] += 1

    allocation = np.zeros((n, n), dtype=int)
    allocation[rows, cols] = base

    return AllocationResult(
        budget=int(budget),
        allocation_matrix=allocation,
        probabilities=probabilities,
        scores_upper=scores_upper,
    )


def _make_rng(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    """Normalize a random-state input to a NumPy Generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)
