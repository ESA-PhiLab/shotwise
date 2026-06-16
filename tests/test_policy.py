"""Tests for shotwise.allocation.policy."""

from __future__ import annotations

import numpy as np
import pytest

from shotwise.allocation.policy import (
    AllocationResult,
    allocate_deterministic_rounding,
    allocate_multinomial,
    normalize_score_matrix,
    upper_triangle_probability_vector,
)


def test_normalize_score_matrix_keeps_strict_upper_triangle() -> None:
    scores = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )

    normalized = normalize_score_matrix(scores)

    expected = np.array(
        [
            [0.0, 2.0, 3.0],
            [0.0, 0.0, 6.0],
            [0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(normalized, expected)


@pytest.mark.parametrize(
    ("scores", "message_part"),
    [
        (np.ones((2, 3)), "square"),
        (np.array([[0.0, -1.0], [0.0, 0.0]]), "negative"),
    ],
)
def test_normalize_score_matrix_rejects_invalid_inputs(
    scores: np.ndarray,
    message_part: str,
) -> None:
    with pytest.raises(ValueError, match=message_part):
        normalize_score_matrix(scores)


def test_upper_triangle_probability_vector_returns_expected_order_and_probs() -> None:
    scores = np.array(
        [
            [0.0, 2.0, 3.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
        ]
    )

    scores_upper, probabilities = upper_triangle_probability_vector(scores)

    expected_scores = np.array([2.0, 3.0, 5.0])
    expected_probabilities = expected_scores / expected_scores.sum()

    np.testing.assert_allclose(scores_upper, expected_scores)
    np.testing.assert_allclose(probabilities, expected_probabilities)
    assert probabilities.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "scores",
    [np.zeros((3, 3)), np.eye(3)],
)
def test_upper_triangle_probability_vector_rejects_zero_total_mass(
    scores: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="strictly positive total mass"):
        upper_triangle_probability_vector(scores)


def test_allocate_multinomial_zero_budget_returns_zero_allocation() -> None:
    scores = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result = allocate_multinomial(scores, budget=0, random_state=123)

    assert isinstance(result, AllocationResult)
    assert result.budget == 0
    np.testing.assert_array_equal(
        result.allocation_matrix,
        np.zeros((3, 3), dtype=int),
    )
    np.testing.assert_array_equal(result.probabilities, np.zeros(3))
    np.testing.assert_array_equal(result.scores_upper, np.zeros(3))


@pytest.mark.parametrize("budget", [-1, -10])
def test_allocate_multinomial_rejects_negative_budget(budget: int) -> None:
    scores = np.array(
        [
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    with pytest.raises(ValueError, match="non-negative"):
        allocate_multinomial(scores, budget=budget)


@pytest.mark.parametrize("scores", [np.zeros((3, 3)), np.eye(3)])
def test_allocate_multinomial_rejects_zero_total_score_for_positive_budget(
    scores: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="strictly positive total mass"):
        allocate_multinomial(scores, budget=5, random_state=0)


def test_allocate_multinomial_is_reproducible_with_seed() -> None:
    scores = np.array(
        [
            [0.0, 2.0, 3.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result_a = allocate_multinomial(scores, budget=50, random_state=42)
    result_b = allocate_multinomial(scores, budget=50, random_state=42)

    np.testing.assert_array_equal(
        result_a.allocation_matrix,
        result_b.allocation_matrix,
    )
    np.testing.assert_allclose(result_a.probabilities, result_b.probabilities)
    np.testing.assert_allclose(result_a.scores_upper, result_b.scores_upper)


def test_allocate_multinomial_preserves_budget_and_upper_triangle() -> None:
    scores = np.array(
        [
            [0.0, 1.0, 4.0],
            [3.0, 0.0, 5.0],
            [2.0, 7.0, 0.0],
        ]
    )

    result = allocate_multinomial(scores, budget=25, random_state=7)

    assert result.budget == 25
    assert result.allocation_matrix.shape == (3, 3)
    assert np.issubdtype(result.allocation_matrix.dtype, np.integer)
    assert result.allocation_matrix.sum() == 25
    assert np.all(np.diag(result.allocation_matrix) == 0)
    assert np.allclose(np.tril(result.allocation_matrix), 0)
    assert np.all(result.probabilities >= 0.0)
    assert result.probabilities.sum() == pytest.approx(1.0)


def test_allocate_deterministic_rounding_zero_budget_returns_zero_allocation() -> None:
    scores = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result = allocate_deterministic_rounding(scores, budget=0)

    assert isinstance(result, AllocationResult)
    assert result.budget == 0
    np.testing.assert_array_equal(
        result.allocation_matrix,
        np.zeros((3, 3), dtype=int),
    )
    np.testing.assert_array_equal(result.probabilities, np.zeros(3))
    np.testing.assert_array_equal(result.scores_upper, np.zeros(3))


@pytest.mark.parametrize("budget", [-1, -10])
def test_allocate_deterministic_rounding_rejects_negative_budget(
    budget: int,
) -> None:
    scores = np.array(
        [
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    with pytest.raises(ValueError, match="non-negative"):
        allocate_deterministic_rounding(scores, budget=budget)


@pytest.mark.parametrize("scores", [np.zeros((3, 3)), np.eye(3)])
def test_allocate_deterministic_rounding_rejects_zero_total_score(
    scores: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="strictly positive total mass"):
        allocate_deterministic_rounding(scores, budget=5)


def test_allocate_deterministic_rounding_matches_floor_plus_remainder_rule() -> None:
    scores = np.array(
        [
            [0.0, 2.0, 3.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result = allocate_deterministic_rounding(scores, budget=11)

    expected_probabilities = np.array([0.2, 0.3, 0.5])
    expected_allocation = np.array(
        [
            [0, 2, 3],
            [0, 0, 6],
            [0, 0, 0],
        ],
        dtype=int,
    )

    np.testing.assert_allclose(result.probabilities, expected_probabilities)
    np.testing.assert_allclose(
        result.scores_upper,
        np.array([2.0, 3.0, 5.0]),
    )
    np.testing.assert_array_equal(
        result.allocation_matrix,
        expected_allocation,
    )
    assert result.allocation_matrix.sum() == 11


def test_allocate_deterministic_rounding_handles_equal_fractional_parts() -> None:
    scores = np.array(
        [
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result = allocate_deterministic_rounding(scores, budget=2)

    # The current implementation breaks ties according to:
    # np.argsort(fractional_parts)[::-1]
    # For equal fractional parts this favors higher flat indices first.
    expected = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=int,
    )

    assert result.allocation_matrix.sum() == 2
    np.testing.assert_array_equal(result.allocation_matrix, expected)
