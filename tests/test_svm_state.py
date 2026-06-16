"""Tests for shotwise.learning.svm."""

from __future__ import annotations

import numpy as np
import pytest

from shotwise.learning.svm import (
    SVMState,
    fit_precomputed_svm_state,
    validate_binary_labels,
)


@pytest.fixture
def separable_kernel() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.9, 0.2, 0.1],
            [0.9, 1.0, 0.1, 0.2],
            [0.2, 0.1, 1.0, 0.9],
            [0.1, 0.2, 0.9, 1.0],
        ]
    )


@pytest.fixture
def labels() -> np.ndarray:
    return np.array([1, 1, -1, -1])


def test_validate_binary_labels_accepts_pm_one_labels() -> None:
    y = np.array([-1, 1, -1, 1])
    result = validate_binary_labels(y)

    np.testing.assert_array_equal(result, y)
    assert result.dtype == int


@pytest.mark.parametrize(
    ("y", "message_part"),
    [
        (np.array([[1, -1], [-1, 1]]), "1D"),
        (np.array([], dtype=int), "must not be empty"),
        (np.array([0, 1, 1]), r"exactly the binary labels -1 and \+1"),
        (np.array([-1, -1, -1]), r"exactly the binary labels -1 and \+1"),
        (np.array([1, 1, 1]), r"exactly the binary labels -1 and \+1"),
    ],
)
def test_validate_binary_labels_rejects_invalid_inputs(
    y: np.ndarray,
    message_part: str,
) -> None:
    with pytest.raises(ValueError, match=message_part):
        validate_binary_labels(y)


def test_fit_precomputed_svm_state_returns_dataclass(
    separable_kernel: np.ndarray,
    labels: np.ndarray,
) -> None:
    state = fit_precomputed_svm_state(separable_kernel, labels, c=1.0)

    assert isinstance(state, SVMState)
    assert state.n_samples == 4
    assert state.num_support >= 1
    assert state.bias == pytest.approx(state.bias)
    assert state.c == pytest.approx(1.0)


def test_fit_precomputed_svm_state_shapes_and_masks(
    separable_kernel: np.ndarray,
    labels: np.ndarray,
) -> None:
    state = fit_precomputed_svm_state(separable_kernel, labels, c=1.0)

    assert state.signed_dual.shape == (4,)
    assert state.alphas.shape == (4,)
    assert state.support_mask.shape == (4,)
    assert state.decision_values.shape == (4,)
    assert state.margin_residuals.shape == (4,)
    assert state.labels.shape == (4,)

    assert state.support_mask.dtype == bool
    np.testing.assert_array_equal(state.active_mask, state.support_mask)
    np.testing.assert_array_equal(
        np.sort(np.flatnonzero(state.support_mask)),
        np.sort(state.support_indices),
    )


def test_alphas_are_non_negative_and_zero_off_support(
    separable_kernel: np.ndarray,
    labels: np.ndarray,
) -> None:
    state = fit_precomputed_svm_state(separable_kernel, labels, c=1.0)

    assert np.all(state.alphas >= -1e-12)
    assert np.allclose(state.alphas[~state.support_mask], 0.0)
    assert np.allclose(state.signed_dual[~state.support_mask], 0.0)

    reconstructed_alphas = state.signed_dual * state.labels
    np.testing.assert_allclose(state.alphas, reconstructed_alphas)


def test_margin_residuals_match_definition(
    separable_kernel: np.ndarray,
    labels: np.ndarray,
) -> None:
    state = fit_precomputed_svm_state(separable_kernel, labels, c=1.0)

    expected = labels * state.decision_values - 1.0
    np.testing.assert_allclose(state.margin_residuals, expected)


def test_support_indices_are_unique_and_in_bounds(
    separable_kernel: np.ndarray,
    labels: np.ndarray,
) -> None:
    state = fit_precomputed_svm_state(separable_kernel, labels, c=1.0)

    assert len(np.unique(state.support_indices)) == state.support_indices.size
    assert np.all(state.support_indices >= 0)
    assert np.all(state.support_indices < state.n_samples)


@pytest.mark.parametrize(
    ("kernel", "y", "c", "message_part"),
    [
        (np.ones((2, 3)), np.array([1, -1]), 1.0, "square"),
        (np.eye(3), np.array([1, -1]), 1.0, "matching leading dimension"),
        (
            np.eye(2),
            np.array([1, 0]),
            1.0,
            r"exactly the binary labels -1 and \+1",
        ),
        (np.eye(2), np.array([1, -1]), 0.0, "strictly positive"),
        (np.eye(2), np.array([1, -1]), -1.0, "strictly positive"),
    ],
)
def test_fit_precomputed_svm_state_rejects_invalid_inputs(
    kernel: np.ndarray,
    y: np.ndarray,
    c: float,
    message_part: str,
) -> None:
    with pytest.raises(ValueError, match=message_part):
        fit_precomputed_svm_state(kernel, y, c=c)


def test_labels_are_copied_in_state(
    separable_kernel: np.ndarray,
    labels: np.ndarray,
) -> None:
    state = fit_precomputed_svm_state(separable_kernel, labels, c=1.0)

    labels[0] = -1
    np.testing.assert_array_equal(state.labels, np.array([1, 1, -1, -1]))
