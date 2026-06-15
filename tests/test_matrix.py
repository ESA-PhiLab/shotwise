"""Tests for shotwise.utils.matrix."""

from __future__ import annotations

import numpy as np
import pytest

from shotwise.utils.matrix import (
    check_square_matrix,
    num_upper_triangle_entries,
    reconstruct_symmetric,
    safe_divide,
    upper_triangle_indices,
    upper_triangle_only,
)


def test_check_square_matrix_accepts_square_matrix() -> None:
    matrix = np.eye(3)
    check_square_matrix(matrix)


@pytest.mark.parametrize(
    ("matrix", "message_part"),
    [
        (np.array([1.0, 2.0, 3.0]), "2D array"),
        (np.ones((2, 3)), "square"),
    ],
)
def test_check_square_matrix_raises_for_invalid_shapes(
    matrix: np.ndarray,
    message_part: str,
) -> None:
    with pytest.raises(ValueError, match=message_part):
        check_square_matrix(matrix, name="test_matrix")


def test_upper_triangle_indices_matches_numpy_convention() -> None:
    rows, cols = upper_triangle_indices(4)

    expected_rows = np.array([0, 0, 0, 1, 1, 2])
    expected_cols = np.array([1, 2, 3, 2, 3, 3])

    np.testing.assert_array_equal(rows, expected_rows)
    np.testing.assert_array_equal(cols, expected_cols)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, 0),
        (1, 0),
        (2, 1),
        (4, 6),
        (5, 10),
    ],
)
def test_num_upper_triangle_entries(n: int, expected: int) -> None:
    assert num_upper_triangle_entries(n) == expected


@pytest.mark.parametrize("n", [-1, -5])
def test_upper_triangle_helpers_reject_negative_dimension(n: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        upper_triangle_indices(n)

    with pytest.raises(ValueError, match="non-negative"):
        num_upper_triangle_entries(n)


def test_upper_triangle_only_keeps_strict_upper_triangle() -> None:
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )

    result = upper_triangle_only(matrix)
    expected = np.array(
        [
            [0.0, 2.0, 3.0],
            [0.0, 0.0, 6.0],
            [0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_array_equal(result, expected)


def test_safe_divide_handles_zero_denominator() -> None:
    numerator = np.array([[2.0, 6.0], [0.0, 5.0]])
    denominator = np.array([[1.0, 2.0], [0.0, 0.0]])

    result = safe_divide(numerator, denominator)
    expected = np.array([[2.0, 3.0], [0.0, 0.0]])

    np.testing.assert_allclose(result, expected)
    assert np.isfinite(result).all()


def test_safe_divide_raises_for_shape_mismatch() -> None:
    numerator = np.ones((2, 2))
    denominator = np.ones((2, 3))

    with pytest.raises(ValueError, match="same shape"):
        safe_divide(numerator, denominator)


def test_reconstruct_symmetric_builds_full_matrix_with_requested_diagonal() -> None:
    upper = np.array(
        [
            [0.0, 0.3, 0.8],
            [9.9, 0.0, 0.5],
            [4.2, 7.7, 0.0],
        ]
    )

    result = reconstruct_symmetric(upper, diagonal_value=1.0)
    expected = np.array(
        [
            [1.0, 0.3, 0.8],
            [0.3, 1.0, 0.5],
            [0.8, 0.5, 1.0],
        ]
    )

    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(result, result.T)


def test_reconstruct_symmetric_ignores_lower_triangle_values() -> None:
    upper = np.array(
        [
            [0.0, 0.1, 0.2],
            [5.0, 0.0, 0.4],
            [6.0, 7.0, 0.0],
        ]
    )

    result = reconstruct_symmetric(upper, diagonal_value=2.0)

    assert result[1, 0] == 0.1
    assert result[2, 0] == 0.2
    assert result[2, 1] == 0.4
    assert np.all(np.diag(result) == 2.0)
