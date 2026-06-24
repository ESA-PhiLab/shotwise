"""Score computation for adaptive shot allocation in ShotWise.

This module implements the score components used by the SupportShot
allocation strategy. The current design mirrors the theory and the original
proof-of-concept implementation while separating the logic into small,
testable functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from shotwise.learning.svm import SVMState
from shotwise.storage.kernel_store import KernelStore
from shotwise.utils.matrix import (
    check_square_matrix,
    upper_triangle_only,
)


@dataclass(frozen=True)
class ScoreComponents:
    """Diagnostics for adaptive allocation score computation.

    Parameters
    ----------
    smoothed_probabilities
        Full symmetric matrix of smoothed Bernoulli probabilities.
    variance_matrix
        Full symmetric matrix of per-entry variance proxies.
    sigma_f
        Per-sample uncertainty proxy induced by kernel variance.
    sigma_safe
        Stabilized version of ``sigma_f`` used for confidence weighting.
    instability_probabilities
        Per-sample probabilities of active-set instability.
    sensitivity_matrix
        Strict upper-triangular raw sensitivity matrix used before normalization.
        In the current formulation, this is the variance-aware sensitivity score.
    instability_matrix
        Strict upper-triangular raw instability matrix used before normalization.
    normalized_sensitivity_matrix
        Strict upper-triangular sensitivity matrix normalized over independent
        upper-triangular entries.
    normalized_instability_matrix
        Strict upper-triangular instability matrix normalized over independent
        upper-triangular entries.
    combined_matrix
        Strict upper-triangular convex combination of the normalized component
        matrices.
    final_scores
        Strict upper-triangular matrix used for shot allocation. This is equal
        to ``combined_matrix`` in the current formulation.
    uncertainty_weight
        Full symmetric matrix ``sqrt(p(1-p))`` used inside the sensitivity term.
    """

    smoothed_probabilities: np.ndarray
    variance_matrix: np.ndarray
    sigma_f: np.ndarray
    sigma_safe: np.ndarray
    instability_probabilities: np.ndarray
    sensitivity_matrix: np.ndarray
    instability_matrix: np.ndarray
    normalized_sensitivity_matrix: np.ndarray
    normalized_instability_matrix: np.ndarray
    combined_matrix: np.ndarray
    final_scores: np.ndarray
    uncertainty_weight: np.ndarray


def compute_smoothed_probabilities(
    counts_matrix: np.ndarray,
    shots_matrix: np.ndarray,
    *,
    prior_success: float = 1.0,
    prior_failure: float = 1.0,
    diagonal_value: float = 1.0,
) -> np.ndarray:
    """Compute smoothed Bernoulli probabilities from counts and shots."""
    counts = np.asarray(counts_matrix, dtype=float)
    shots = np.asarray(shots_matrix, dtype=float)
    check_square_matrix(counts, name="counts_matrix")
    check_square_matrix(shots, name="shots_matrix")

    if counts.shape != shots.shape:
        raise ValueError(
            "counts_matrix and shots_matrix must have the same shape, "
            f"got {counts.shape} and {shots.shape}."
        )

    if prior_success < 0.0 or prior_failure < 0.0:
        raise ValueError("Beta prior pseudocounts must be non-negative.")

    probs = np.full_like(counts, 0.5, dtype=float)
    mask = shots > 0
    denominator = shots + prior_success + prior_failure
    probs[mask] = (counts[mask] + prior_success) / denominator[mask]
    np.fill_diagonal(probs, diagonal_value)
    return probs


def compute_variance_matrix(
    smoothed_probabilities: np.ndarray,
    shots_matrix: np.ndarray,
    *,
    hardware_floor: float = 0.0,
) -> np.ndarray:
    """Compute a full symmetric kernel variance proxy."""
    probs = np.asarray(smoothed_probabilities, dtype=float)
    shots = np.asarray(shots_matrix, dtype=float)
    check_square_matrix(probs, name="smoothed_probabilities")
    check_square_matrix(shots, name="shots_matrix")

    if probs.shape != shots.shape:
        raise ValueError(
            "smoothed_probabilities and shots_matrix must have the same "
            f"shape, got {probs.shape} and {shots.shape}."
        )

    if hardware_floor < 0.0:
        raise ValueError(f"hardware_floor must be non-negative, got {hardware_floor}.")

    variance = np.zeros_like(probs, dtype=float)
    mask = shots > 0
    variance[mask] = probs[mask] * (1.0 - probs[mask]) / shots[mask]
    if hardware_floor > 0.0:
        variance[mask] += hardware_floor
    np.fill_diagonal(variance, 0.0)
    return variance


def compute_sigma_f(
    state: SVMState,
    variance_matrix: np.ndarray,
) -> np.ndarray:
    """Compute per-sample uncertainty induced by kernel variance."""
    variance = np.asarray(variance_matrix, dtype=float)
    check_square_matrix(variance, name="variance_matrix")

    if variance.shape[0] != state.n_samples:
        raise ValueError(
            "variance_matrix dimension must match the SVM state size. "
            f"Got variance shape {variance.shape} and n={state.n_samples}."
        )

    alpha_sq = state.alphas**2
    sigma_sq = variance @ alpha_sq
    sigma_sq = np.clip(sigma_sq, a_min=0.0, a_max=None)
    return np.sqrt(sigma_sq)


def compute_instability_probabilities(
    state: SVMState,
    sigma_f: np.ndarray,
    *,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Compute probabilities of active-set instability."""
    sigma = np.asarray(sigma_f, dtype=float)
    if sigma.shape != (state.n_samples,):
        raise ValueError(
            f"sigma_f must have shape ({state.n_samples},), got {sigma.shape}."
        )
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be strictly positive, got {epsilon}.")

    return norm.cdf(-state.margin_residuals / (sigma + epsilon))


def compute_sensitivity_matrix(
    state: SVMState,
    *,
    sigma_safe: np.ndarray | None = None,
    use_confidence_weighting: bool = False,
    uncertainty_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the strict upper-triangular sensitivity score matrix.

    The base geometric sensitivity is ``|alpha_i alpha_j|``. If
    ``use_confidence_weighting`` is enabled, the dual coefficients are first
    scaled by ``sigma_safe``. If ``uncertainty_weight`` is provided, it is
    multiplied entrywise into the resulting pairwise matrix so that the final
    sensitivity score is variance-aware.
    """
    if sigma_safe is None:
        effective_alpha = state.alphas
    else:
        sigma_safe = np.asarray(sigma_safe, dtype=float)
        if sigma_safe.shape != (state.n_samples,):
            raise ValueError(
                f"sigma_safe must have shape ({state.n_samples},), "
                f"got {sigma_safe.shape}."
            )
        if np.any(sigma_safe <= 0.0):
            raise ValueError("sigma_safe must be strictly positive.")
        effective_alpha = (
            state.alphas / sigma_safe if use_confidence_weighting else state.alphas
        )

    sensitivity = np.abs(np.outer(effective_alpha, effective_alpha))

    if uncertainty_weight is not None:
        weight = np.asarray(uncertainty_weight, dtype=float)
        check_square_matrix(weight, name="uncertainty_weight")
        if weight.shape != sensitivity.shape:
            raise ValueError(
                "uncertainty_weight must have the same shape as the sensitivity "
                f"matrix, got {weight.shape} and {sensitivity.shape}."
            )
        sensitivity = sensitivity * weight

    return upper_triangle_only(sensitivity)


def compute_instability_matrix(
    instability_probabilities: np.ndarray,
    *,
    c_value: float,
) -> np.ndarray:
    """Compute the strict upper-triangular instability score matrix."""
    probs = np.asarray(instability_probabilities, dtype=float)
    if probs.ndim != 1:
        raise ValueError(
            f"instability_probabilities must be 1D, got shape={probs.shape}."
        )
    if c_value <= 0.0:
        raise ValueError(f"c_value must be strictly positive, got {c_value}.")

    instability = (c_value**2) * np.outer(probs, probs)
    return upper_triangle_only(instability)


def normalize_score_matrix(score_matrix: np.ndarray) -> np.ndarray:
    """Normalize a strict upper-triangular score matrix over independent entries.

    If the matrix has zero total mass in the strict upper triangle, returns
    the zero matrix.
    """
    matrix = np.asarray(score_matrix, dtype=float)
    check_square_matrix(matrix, name="score_matrix")

    matrix = upper_triangle_only(matrix)
    total = float(matrix.sum())

    if total <= 0.0:
        return np.zeros_like(matrix)

    return matrix / total


def combine_score_matrices(
    sensitivity_matrix: np.ndarray,
    instability_matrix: np.ndarray,
    *,
    lambda_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize component matrices separately and combine them with lambda.

    Parameters
    ----------
    sensitivity_matrix
        Strict upper-triangular raw sensitivity score matrix.
    instability_matrix
        Strict upper-triangular raw instability score matrix.
    lambda_weight
        Convex mixing parameter in ``[0, 1]``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(normalized_sensitivity, normalized_instability, combined_matrix)``
        where all matrices are strict upper-triangular.
    """
    sensitivity = np.asarray(sensitivity_matrix, dtype=float)
    instability = np.asarray(instability_matrix, dtype=float)
    check_square_matrix(sensitivity, name="sensitivity_matrix")
    check_square_matrix(instability, name="instability_matrix")

    if sensitivity.shape != instability.shape:
        raise ValueError(
            "sensitivity_matrix and instability_matrix must have the same "
            f"shape, got {sensitivity.shape} and {instability.shape}."
        )

    if not 0.0 <= lambda_weight <= 1.0:
        raise ValueError(f"lambda_weight must lie in [0, 1], got {lambda_weight}.")

    normalized_sensitivity = normalize_score_matrix(sensitivity)
    normalized_instability = normalize_score_matrix(instability)

    combined = (1.0 - lambda_weight) * normalized_sensitivity + (
        lambda_weight * normalized_instability
    )
    combined = upper_triangle_only(combined)

    return normalized_sensitivity, normalized_instability, combined


def compute_supportshot_scores(
    store: KernelStore,
    state: SVMState,
    *,
    lambda_weight: float = 0.5,
    use_confidence_weighting: bool = False,
    sigma_floor: float = 1e-8,
    hardware_floor: float = 0.0,
    prior_success: float = 1.0,
    prior_failure: float = 1.0,
    epsilon: float = 1e-9,
) -> ScoreComponents:
    """Compute SupportShot allocation scores from the current store/state.

    The updated score pipeline is:

    1. smooth empirical probabilities,
    2. compute kernel variance proxy,
    3. compute per-sample uncertainty ``sigma_f``,
    4. estimate instability probabilities,
    5. compute a variance-aware sensitivity matrix,
    6. compute an instability matrix,
    7. normalize both matrices separately over the strict upper triangle,
    8. combine them using ``lambda_weight``.
    """
    if sigma_floor < 0.0:
        raise ValueError(f"sigma_floor must be non-negative, got {sigma_floor}.")

    counts_matrix = store.counts_matrix()
    shots_matrix = store.shots_matrix()

    smoothed_probabilities = compute_smoothed_probabilities(
        counts_matrix,
        shots_matrix,
        prior_success=prior_success,
        prior_failure=prior_failure,
        diagonal_value=store.diagonal_value,
    )
    variance_matrix = compute_variance_matrix(
        smoothed_probabilities,
        shots_matrix,
        hardware_floor=hardware_floor,
    )
    sigma_f = compute_sigma_f(state, variance_matrix)
    sigma_safe = np.sqrt(sigma_f**2 + sigma_floor)
    instability_probabilities = compute_instability_probabilities(
        state,
        sigma_f,
        epsilon=epsilon,
    )

    uncertainty_weight = np.sqrt(
        np.clip(
            smoothed_probabilities * (1.0 - smoothed_probabilities),
            a_min=0.0,
            a_max=None,
        )
    )
    np.fill_diagonal(uncertainty_weight, 0.0)

    sensitivity_matrix = compute_sensitivity_matrix(
        state,
        sigma_safe=sigma_safe,
        use_confidence_weighting=use_confidence_weighting,
        uncertainty_weight=uncertainty_weight,
    )
    instability_matrix = compute_instability_matrix(
        instability_probabilities,
        c_value=state.c,
    )

    (
        normalized_sensitivity_matrix,
        normalized_instability_matrix,
        combined_matrix,
    ) = combine_score_matrices(
        sensitivity_matrix,
        instability_matrix,
        lambda_weight=lambda_weight,
    )

    final_scores = combined_matrix.copy()

    return ScoreComponents(
        smoothed_probabilities=smoothed_probabilities,
        variance_matrix=variance_matrix,
        sigma_f=sigma_f,
        sigma_safe=sigma_safe,
        instability_probabilities=instability_probabilities,
        sensitivity_matrix=sensitivity_matrix,
        instability_matrix=instability_matrix,
        normalized_sensitivity_matrix=normalized_sensitivity_matrix,
        normalized_instability_matrix=normalized_instability_matrix,
        combined_matrix=combined_matrix,
        final_scores=final_scores,
        uncertainty_weight=uncertainty_weight,
    )
