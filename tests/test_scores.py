"""Tests for shotwise.allocation.scores."""

from __future__ import annotations

from math import erf, sqrt

import numpy as np
import pytest

from shotwise.allocation.scores import (
    ScoreComponents,
    combine_score_matrices,
    compute_instability_matrix,
    compute_instability_probabilities,
    compute_sensitivity_matrix,
    compute_sigma_f,
    compute_smoothed_probabilities,
    compute_supportshot_scores,
    compute_variance_matrix,
)
from shotwise.learning.svm import SVMState
from shotwise.storage.kernel_store import KernelStore


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


@pytest.fixture
def svm_state() -> SVMState:
    labels = np.array([1, -1, 1])
    alphas = np.array([1.0, 2.0, 0.0])
    signed_dual = alphas * labels
    support_indices = np.array([0, 1])
    support_mask = np.array([True, True, False])
    decision_values = np.array([1.5, -1.2, 0.4])
    margin_residuals = labels * decision_values - 1.0

    return SVMState(
        signed_dual=signed_dual,
        alphas=alphas,
        support_indices=support_indices,
        support_mask=support_mask,
        decision_values=decision_values,
        margin_residuals=margin_residuals,
        bias=0.1,
        labels=labels,
        c=1.5,
    )


@pytest.fixture
def store() -> KernelStore:
    kernel_store = KernelStore(3, diagonal_value=1.0)
    counts = np.array(
        [
            [0, 2, 1],
            [0, 0, 3],
            [0, 0, 0],
        ],
        dtype=int,
    )
    shots = np.array(
        [
            [0, 4, 2],
            [0, 0, 6],
            [0, 0, 0],
        ],
        dtype=int,
    )
    kernel_store.update(counts, shots)
    return kernel_store


def test_compute_smoothed_probabilities_basic() -> None:
    counts = np.array(
        [
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 3.0],
            [1.0, 3.0, 0.0],
        ]
    )
    shots = np.array(
        [
            [0.0, 4.0, 2.0],
            [4.0, 0.0, 6.0],
            [2.0, 6.0, 0.0],
        ]
    )

    probs = compute_smoothed_probabilities(counts, shots, diagonal_value=1.0)

    expected = np.array(
        [
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ]
    )
    np.testing.assert_allclose(probs, expected)


@pytest.mark.parametrize(
    ("counts", "shots", "message_part"),
    [
        (np.ones((2, 3)), np.ones((2, 3)), "square"),
        (np.ones((2, 2)), np.ones((3, 3)), "same shape"),
    ],
)
def test_compute_smoothed_probabilities_rejects_invalid_shapes(
    counts: np.ndarray,
    shots: np.ndarray,
    message_part: str,
) -> None:
    with pytest.raises(ValueError, match=message_part):
        compute_smoothed_probabilities(counts, shots)


@pytest.mark.parametrize(
    ("prior_success", "prior_failure"),
    [(-1.0, 1.0), (1.0, -1.0)],
)
def test_compute_smoothed_probabilities_rejects_negative_priors(
    prior_success: float,
    prior_failure: float,
) -> None:
    counts = np.zeros((2, 2))
    shots = np.zeros((2, 2))
    with pytest.raises(ValueError, match="non-negative"):
        compute_smoothed_probabilities(
            counts,
            shots,
            prior_success=prior_success,
            prior_failure=prior_failure,
        )


def test_compute_variance_matrix_basic_and_hardware_floor() -> None:
    probs = np.array(
        [
            [1.0, 0.5, 0.25],
            [0.5, 1.0, 0.75],
            [0.25, 0.75, 1.0],
        ]
    )
    shots = np.array(
        [
            [0.0, 4.0, 2.0],
            [4.0, 0.0, 8.0],
            [2.0, 8.0, 0.0],
        ]
    )

    variance = compute_variance_matrix(probs, shots, hardware_floor=0.01)

    expected = np.array(
        [
            [0.0, 0.0725, 0.10375],
            [0.0725, 0.0, 0.0334375],
            [0.10375, 0.0334375, 0.0],
        ]
    )
    np.testing.assert_allclose(variance, expected)


@pytest.mark.parametrize(
    ("probs", "shots", "hardware_floor", "message_part"),
    [
        (np.ones((2, 3)), np.ones((2, 3)), 0.0, "square"),
        (np.eye(2), np.eye(3), 0.0, "same shape"),
        (np.eye(2), np.eye(2), -0.1, "non-negative"),
    ],
)
def test_compute_variance_matrix_rejects_invalid_inputs(
    probs: np.ndarray,
    shots: np.ndarray,
    hardware_floor: float,
    message_part: str,
) -> None:
    with pytest.raises(ValueError, match=message_part):
        compute_variance_matrix(probs, shots, hardware_floor=hardware_floor)


def test_compute_sigma_f_matches_manual_formula(svm_state: SVMState) -> None:
    variance = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.1, 0.0, 0.3],
            [0.2, 0.3, 0.0],
        ]
    )

    sigma_f = compute_sigma_f(svm_state, variance)

    # alphas^2 = [1, 4, 0]
    expected = np.sqrt(np.array([0.4, 0.1, 1.4]))
    np.testing.assert_allclose(sigma_f, expected)


def test_compute_sigma_f_rejects_invalid_shape(svm_state: SVMState) -> None:
    with pytest.raises(ValueError, match="dimension must match"):
        compute_sigma_f(svm_state, np.eye(2))


def test_compute_instability_probabilities_matches_formula(
    svm_state: SVMState,
) -> None:
    sigma_f = np.array([0.2, 0.3, 0.4])
    probs = compute_instability_probabilities(svm_state, sigma_f, epsilon=1e-9)

    expected = _normal_cdf(-svm_state.margin_residuals / (sigma_f + 1e-9))
    np.testing.assert_allclose(probs, expected)


@pytest.mark.parametrize(
    ("sigma_f", "epsilon", "message_part"),
    [
        (np.array([0.1, 0.2]), 1e-9, "shape"),
        (np.array([0.1, 0.2, 0.3]), 0.0, "strictly positive"),
    ],
)
def test_compute_instability_probabilities_rejects_invalid_inputs(
    svm_state: SVMState,
    sigma_f: np.ndarray,
    epsilon: float,
    message_part: str,
) -> None:
    with pytest.raises(ValueError, match=message_part):
        compute_instability_probabilities(svm_state, sigma_f, epsilon=epsilon)


def test_compute_sensitivity_matrix_without_confidence_weighting(
    svm_state: SVMState,
) -> None:
    sensitivity = compute_sensitivity_matrix(svm_state)
    expected = np.array(
        [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(sensitivity, expected)


def test_compute_sensitivity_matrix_with_confidence_weighting(
    svm_state: SVMState,
) -> None:
    sigma_safe = np.array([2.0, 4.0, 1.0])
    sensitivity = compute_sensitivity_matrix(
        svm_state,
        sigma_safe=sigma_safe,
        use_confidence_weighting=True,
    )
    expected = np.array(
        [
            [0.0, 0.25, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(sensitivity, expected)


@pytest.mark.parametrize(
    ("sigma_safe", "message_part"),
    [
        (np.array([1.0, 2.0]), "shape"),
        (np.array([1.0, 0.0, 1.0]), "strictly positive"),
    ],
)
def test_compute_sensitivity_matrix_rejects_invalid_sigma_safe(
    svm_state: SVMState,
    sigma_safe: np.ndarray,
    message_part: str,
) -> None:
    with pytest.raises(ValueError, match=message_part):
        compute_sensitivity_matrix(
            svm_state,
            sigma_safe=sigma_safe,
            use_confidence_weighting=True,
        )


def test_compute_instability_matrix_basic(svm_state: SVMState) -> None:
    probs = np.array([0.1, 0.2, 0.3])
    instability = compute_instability_matrix(probs, c_value=svm_state.c)
    expected = np.array(
        [
            [0.0, 0.045, 0.0675],
            [0.0, 0.0, 0.135],
            [0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(instability, expected)


@pytest.mark.parametrize(
    ("probs", "c_value", "message_part"),
    [
        (np.ones((2, 2)), 1.0, "1D"),
        (np.array([0.1, 0.2]), 0.0, "strictly positive"),
    ],
)
def test_compute_instability_matrix_rejects_invalid_inputs(
    probs: np.ndarray,
    c_value: float,
    message_part: str,
) -> None:
    with pytest.raises(ValueError, match=message_part):
        compute_instability_matrix(probs, c_value=c_value)


def test_combine_score_matrices_basic() -> None:
    sensitivity = np.array(
        [
            [0.0, 2.0, 1.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 0.0],
        ]
    )
    instability = np.array(
        [
            [0.0, 1.0, 3.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
        ]
    )
    weight = np.array(
        [
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 2.0],
            [1.0, 2.0, 0.0],
        ]
    )

    combined, final_scores = combine_score_matrices(
        sensitivity,
        instability,
        lambda_weight=0.25,
        uncertainty_weight=weight,
    )

    expected_combined = np.array(
        [
            [0.0, 1.75, 1.5],
            [0.0, 0.0, 4.25],
            [0.0, 0.0, 0.0],
        ]
    )
    expected_final = np.array(
        [
            [0.0, 0.875, 1.5],
            [0.0, 0.0, 8.5],
            [0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(combined, expected_combined)
    np.testing.assert_allclose(final_scores, expected_final)


@pytest.mark.parametrize(
    ("sensitivity", "instability", "lambda_weight", "weight", "message_part"),
    [
        (np.ones((2, 3)), np.ones((2, 3)), 0.5, None, "square"),
        (np.eye(2), np.eye(3), 0.5, None, "same shape"),
        (np.eye(2), np.eye(2), -0.1, None, r"lie in \[0, 1\]"),
        (np.eye(2), np.eye(2), 0.5, np.ones((3, 3)), "same shape"),
    ],
)
def test_combine_score_matrices_rejects_invalid_inputs(
    sensitivity: np.ndarray,
    instability: np.ndarray,
    lambda_weight: float,
    weight: np.ndarray | None,
    message_part: str,
) -> None:
    with pytest.raises(ValueError, match=message_part):
        combine_score_matrices(
            sensitivity,
            instability,
            lambda_weight=lambda_weight,
            uncertainty_weight=weight,
        )


def test_compute_supportshot_scores_end_to_end(
    store: KernelStore,
    svm_state: SVMState,
) -> None:
    components = compute_supportshot_scores(
        store,
        svm_state,
        lambda_weight=0.5,
        use_confidence_weighting=True,
        sigma_floor=1e-6,
        hardware_floor=0.01,
    )

    assert isinstance(components, ScoreComponents)
    assert components.smoothed_probabilities.shape == (3, 3)
    assert components.variance_matrix.shape == (3, 3)
    assert components.sigma_f.shape == (3,)
    assert components.sigma_safe.shape == (3,)
    assert components.instability_probabilities.shape == (3,)
    assert components.sensitivity_matrix.shape == (3, 3)
    assert components.instability_matrix.shape == (3, 3)
    assert components.combined_matrix.shape == (3, 3)
    assert components.final_scores.shape == (3, 3)
    assert components.uncertainty_weight.shape == (3, 3)

    assert np.all(np.diag(components.smoothed_probabilities) == store.diagonal_value)
    assert np.all(np.diag(components.variance_matrix) == 0.0)
    assert np.all(np.diag(components.uncertainty_weight) == 0.0)
    assert np.allclose(np.tril(components.final_scores), 0.0)
    assert np.all(components.final_scores >= 0.0)
    assert np.all(components.sigma_safe > 0.0)


def test_compute_supportshot_scores_rejects_negative_sigma_floor(
    store: KernelStore,
    svm_state: SVMState,
) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compute_supportshot_scores(store, svm_state, sigma_floor=-1e-3)
