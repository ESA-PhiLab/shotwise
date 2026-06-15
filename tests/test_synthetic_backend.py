"""Tests for shotwise.backends.synthetic."""

from __future__ import annotations

import numpy as np
import pytest

from shotwise.backends.synthetic import (
    MeasurementEstimate,
    SyntheticBernoulliBackend,
)


def test_measurement_estimate_dataclass_fields() -> None:
    estimate = MeasurementEstimate(
        i=0,
        j=1,
        shots=10,
        successes=7,
        mean=0.7,
        probability_used=0.75,
    )

    assert estimate.i == 0
    assert estimate.j == 1
    assert estimate.shots == 10
    assert estimate.successes == 7
    assert estimate.mean == 0.7
    assert estimate.probability_used == 0.75


@pytest.fixture
def kernel_matrix() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.5],
            [0.2, 0.5, 1.0],
        ]
    )


def test_backend_rejects_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="square"):
        SyntheticBernoulliBackend(np.ones((2, 3)))


def test_backend_rejects_non_symmetric_matrix() -> None:
    matrix = np.array(
        [
            [1.0, 0.2],
            [0.7, 1.0],
        ]
    )
    with pytest.raises(ValueError, match="symmetric"):
        SyntheticBernoulliBackend(matrix)


@pytest.mark.parametrize("bad_value", [-0.1, 1.1])
def test_backend_rejects_probabilities_outside_unit_interval(
    bad_value: float,
) -> None:
    matrix = np.array(
        [
            [1.0, bad_value],
            [bad_value, 1.0],
        ]
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SyntheticBernoulliBackend(matrix)


@pytest.mark.parametrize(
    ("kwargs", "message_part"),
    [
        ({"noise_sigma_global": -0.1}, "noise_sigma_global"),
        ({"noise_sigma_local": -0.2}, "noise_sigma_local"),
    ],
)
def test_backend_rejects_negative_noise_sigmas(
    kernel_matrix: np.ndarray,
    kwargs: dict[str, float],
    message_part: str,
) -> None:
    with pytest.raises(ValueError, match=message_part):
        SyntheticBernoulliBackend(kernel_matrix, **kwargs)


def test_kernel_matrix_property_returns_copy(kernel_matrix: np.ndarray) -> None:
    backend = SyntheticBernoulliBackend(kernel_matrix, random_state=0)

    returned = backend.kernel_matrix
    returned[0, 1] = 0.0

    np.testing.assert_allclose(backend.kernel_matrix, kernel_matrix)


def test_estimate_entries_returns_empty_list_for_no_pairs(
    kernel_matrix: np.ndarray,
) -> None:
    backend = SyntheticBernoulliBackend(kernel_matrix, random_state=0)

    estimates = backend.estimate_entries([], [])

    assert estimates == []


def test_estimate_entries_normalizes_pair_order(
    kernel_matrix: np.ndarray,
) -> None:
    backend = SyntheticBernoulliBackend(kernel_matrix, random_state=0)

    estimates = backend.estimate_entries([(2, 0)], 5)

    assert len(estimates) == 1
    estimate = estimates[0]
    assert estimate.i == 0
    assert estimate.j == 2


@pytest.mark.parametrize("pair", [(0, 0), (1, 1), (2, 2)])
def test_estimate_entries_rejects_diagonal_pairs(
    kernel_matrix: np.ndarray,
    pair: tuple[int, int],
) -> None:
    backend = SyntheticBernoulliBackend(kernel_matrix, random_state=0)

    with pytest.raises(ValueError, match="Diagonal"):
        backend.estimate_entries([pair], 10)


@pytest.mark.parametrize("pair", [(-1, 1), (0, 3), (4, 1)])
def test_estimate_entries_rejects_out_of_bounds_pairs(
    kernel_matrix: np.ndarray,
    pair: tuple[int, int],
) -> None:
    backend = SyntheticBernoulliBackend(kernel_matrix, random_state=0)

    with pytest.raises(ValueError, match="out of bounds"):
        backend.estimate_entries([pair], 10)


@pytest.mark.parametrize("shots", [-1, [-1]])
def test_estimate_entries_rejects_negative_shots(
    kernel_matrix: np.ndarray,
    shots: int | list[int],
) -> None:
    backend = SyntheticBernoulliBackend(kernel_matrix, random_state=0)

    with pytest.raises(ValueError, match="non-negative"):
        backend.estimate_entries([(0, 1)], shots)


def test_estimate_entries_rejects_mismatched_shot_sequence_length(
    kernel_matrix: np.ndarray,
) -> None:
    backend = SyntheticBernoulliBackend(kernel_matrix, random_state=0)

    with pytest.raises(ValueError, match="same length as pairs"):
        backend.estimate_entries([(0, 1), (0, 2)], [10])


def test_estimate_entries_with_zero_shots_returns_zero_mean(
    kernel_matrix: np.ndarray,
) -> None:
    backend = SyntheticBernoulliBackend(kernel_matrix, random_state=0)

    estimate = backend.estimate_entries([(0, 1)], 0)[0]

    assert estimate.shots == 0
    assert estimate.successes == 0
    assert estimate.mean == 0.0


def test_zero_noise_probabilities_match_true_kernel(
    kernel_matrix: np.ndarray,
) -> None:
    backend = SyntheticBernoulliBackend(
        kernel_matrix,
        noise_sigma_global=0.0,
        noise_sigma_local=0.0,
        random_state=0,
    )

    estimates = backend.estimate_entries([(0, 1), (0, 2), (1, 2)], [5, 6, 7])
    probabilities = [estimate.probability_used for estimate in estimates]

    np.testing.assert_allclose(probabilities, [0.8, 0.2, 0.5])


def test_estimate_entries_is_reproducible_for_same_seed(
    kernel_matrix: np.ndarray,
) -> None:
    backend_a = SyntheticBernoulliBackend(kernel_matrix, random_state=123)
    backend_b = SyntheticBernoulliBackend(kernel_matrix, random_state=123)

    estimates_a = backend_a.estimate_entries([(0, 1), (0, 2)], [50, 60])
    estimates_b = backend_b.estimate_entries([(0, 1), (0, 2)], [50, 60])

    assert estimates_a == estimates_b


def test_sample_upper_triangle_counts_returns_upper_triangle_only(
    kernel_matrix: np.ndarray,
) -> None:
    backend = SyntheticBernoulliBackend(kernel_matrix, random_state=0)

    shot_matrix = np.array(
        [
            [99, 10, 20],
            [30, 88, 15],
            [40, 50, 77],
        ]
    )

    counts = backend.sample_upper_triangle_counts(shot_matrix)

    assert counts.shape == shot_matrix.shape
    assert np.all(np.diag(counts) == 0)
    assert np.allclose(np.tril(counts), 0)
    assert counts[0, 1] <= 10
    assert counts[0, 2] <= 20
    assert counts[1, 2] <= 15


def test_sample_upper_triangle_counts_rejects_wrong_shape(
    kernel_matrix: np.ndarray,
) -> None:
    backend = SyntheticBernoulliBackend(kernel_matrix, random_state=0)

    with pytest.raises(ValueError, match="incompatible shape"):
        backend.sample_upper_triangle_counts(np.ones((4, 4), dtype=int))


def test_sample_upper_triangle_counts_rejects_negative_shots(
    kernel_matrix: np.ndarray,
) -> None:
    backend = SyntheticBernoulliBackend(kernel_matrix, random_state=0)

    shot_matrix = np.array(
        [
            [0, -1, 2],
            [0, 0, 3],
            [0, 0, 0],
        ]
    )

    with pytest.raises(ValueError, match="negative"):
        backend.sample_upper_triangle_counts(shot_matrix)


def test_sample_upper_triangle_counts_reproducible_with_same_seed(
    kernel_matrix: np.ndarray,
) -> None:
    shot_matrix = np.array(
        [
            [0, 10, 20],
            [0, 0, 15],
            [0, 0, 0],
        ]
    )

    backend_a = SyntheticBernoulliBackend(kernel_matrix, random_state=42)
    backend_b = SyntheticBernoulliBackend(kernel_matrix, random_state=42)

    counts_a = backend_a.sample_upper_triangle_counts(shot_matrix)
    counts_b = backend_b.sample_upper_triangle_counts(shot_matrix)

    np.testing.assert_array_equal(counts_a, counts_b)


def test_nonzero_noise_clips_probabilities_to_unit_interval() -> None:
    kernel_matrix = np.array(
        [
            [1.0, 0.99],
            [0.99, 1.0],
        ]
    )
    backend = SyntheticBernoulliBackend(
        kernel_matrix,
        noise_sigma_global=10.0,
        noise_sigma_local=10.0,
        random_state=0,
    )

    estimate = backend.estimate_entries([(0, 1)], 1)[0]

    assert 0.0 <= estimate.probability_used <= 1.0
