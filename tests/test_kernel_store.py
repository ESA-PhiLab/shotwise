"""Tests for shotwise.storage.kernel_store."""

from __future__ import annotations

import numpy as np
import pytest

from shotwise.storage.kernel_store import KernelStore


def test_init_rejects_non_positive_dimension() -> None:
    with pytest.raises(ValueError, match="positive"):
        KernelStore(0)

    with pytest.raises(ValueError, match="positive"):
        KernelStore(-3)


def test_init_starts_with_zero_state() -> None:
    store = KernelStore(3, diagonal_value=1.0)

    np.testing.assert_array_equal(store.counts_upper, np.zeros((3, 3), dtype=int))
    np.testing.assert_array_equal(store.shots_upper, np.zeros((3, 3), dtype=int))
    assert store.total_shots_spent() == 0
    assert store.num_measured_entries() == 0


def test_properties_return_copies() -> None:
    store = KernelStore(3)

    counts = store.counts_upper
    shots = store.shots_upper
    counts[0, 1] = 7
    shots[0, 1] = 9

    assert store.counts_upper[0, 1] == 0
    assert store.shots_upper[0, 1] == 0


def test_update_accumulates_counts_and_shots() -> None:
    store = KernelStore(3, diagonal_value=1.0)

    counts_a = np.array(
        [
            [0, 2, 1],
            [0, 0, 3],
            [0, 0, 0],
        ],
        dtype=int,
    )
    shots_a = np.array(
        [
            [0, 5, 4],
            [0, 0, 6],
            [0, 0, 0],
        ],
        dtype=int,
    )

    counts_b = np.array(
        [
            [0, 1, 2],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=int,
    )
    shots_b = np.array(
        [
            [0, 2, 3],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=int,
    )

    store.update(counts_a, shots_a)
    store.update(counts_b, shots_b)

    expected_counts = np.array(
        [
            [0, 3, 3],
            [0, 0, 3],
            [0, 0, 0],
        ],
        dtype=int,
    )
    expected_shots = np.array(
        [
            [0, 7, 7],
            [0, 0, 6],
            [0, 0, 0],
        ],
        dtype=int,
    )

    np.testing.assert_array_equal(store.counts_upper, expected_counts)
    np.testing.assert_array_equal(store.shots_upper, expected_shots)
    assert store.total_shots_spent() == 20
    assert store.num_measured_entries() == 3


@pytest.mark.parametrize(
    ("counts", "shots", "message_part"),
    [
        (
            np.zeros((2, 2), dtype=int),
            np.zeros((3, 3), dtype=int),
            "shape",
        ),
        (
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=int),
            np.array([[0, 1], [0, 0]], dtype=int),
            "shape",
        ),
    ],
)
def test_update_rejects_wrong_shape(
    counts: np.ndarray,
    shots: np.ndarray,
    message_part: str,
) -> None:
    store = KernelStore(3)

    with pytest.raises(ValueError, match=message_part):
        store.update(counts, shots)


@pytest.mark.parametrize(
    ("counts", "shots", "message_part"),
    [
        (
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=int),
            np.zeros((3, 3), dtype=int),
            "strict upper triangle",
        ),
        (
            np.array([[0, 1, 0], [9, 0, 0], [0, 0, 0]], dtype=int),
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=int),
            "strict upper triangle",
        ),
        (
            np.zeros((3, 3), dtype=int),
            np.array([[0, 1, 0], [4, 0, 0], [0, 0, 0]], dtype=int),
            "strict upper triangle",
        ),
    ],
)
def test_update_rejects_non_upper_triangle_entries(
    counts: np.ndarray,
    shots: np.ndarray,
    message_part: str,
) -> None:
    store = KernelStore(3)

    with pytest.raises(ValueError, match=message_part):
        store.update(counts, shots)


@pytest.mark.parametrize(
    ("counts", "shots", "message_part"),
    [
        (
            np.array([[0, -1, 0], [0, 0, 0], [0, 0, 0]], dtype=int),
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=int),
            "negative",
        ),
        (
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=int),
            np.array([[0, -1, 0], [0, 0, 0], [0, 0, 0]], dtype=int),
            "negative",
        ),
    ],
)
def test_update_rejects_negative_values(
    counts: np.ndarray,
    shots: np.ndarray,
    message_part: str,
) -> None:
    store = KernelStore(3)

    with pytest.raises(ValueError, match=message_part):
        store.update(counts, shots)


def test_update_rejects_counts_exceeding_shots() -> None:
    store = KernelStore(3)

    counts = np.array(
        [
            [0, 5, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=int,
    )
    shots = np.array(
        [
            [0, 4, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=int,
    )

    with pytest.raises(ValueError, match="must not exceed"):
        store.update(counts, shots)


def test_update_accepts_integer_valued_float_inputs() -> None:
    store = KernelStore(3)

    counts = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    shots = np.array(
        [
            [0.0, 3.0, 4.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    store.update(counts, shots)

    np.testing.assert_array_equal(
        store.counts_upper,
        np.array(
            [
                [0, 1, 2],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=int,
        ),
    )
    np.testing.assert_array_equal(
        store.shots_upper,
        np.array(
            [
                [0, 3, 4],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=int,
        ),
    )


@pytest.mark.parametrize(
    ("counts", "shots"),
    [
        (
            np.array([[0.0, 1.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[0, 2, 0], [0, 0, 0], [0, 0, 0]], dtype=int),
        ),
        (
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=int),
            np.array([[0.0, 2.2, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ),
    ],
)
def test_update_rejects_non_integer_float_inputs(
    counts: np.ndarray,
    shots: np.ndarray,
) -> None:
    store = KernelStore(3)

    with pytest.raises(ValueError, match="integer values"):
        store.update(counts, shots)


def test_kernel_upper_and_kernel_matrix() -> None:
    store = KernelStore(3, diagonal_value=1.0)

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
    store.update(counts, shots)

    expected_upper = np.array(
        [
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
        ]
    )
    expected_kernel = np.array(
        [
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ]
    )

    np.testing.assert_allclose(store.kernel_upper(), expected_upper)
    np.testing.assert_allclose(store.kernel_matrix(), expected_kernel)


def test_counts_and_shots_matrix_are_symmetric() -> None:
    store = KernelStore(3)

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
            [0, 5, 4],
            [0, 0, 6],
            [0, 0, 0],
        ],
        dtype=int,
    )
    store.update(counts, shots)

    expected_counts_matrix = np.array(
        [
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 3.0],
            [1.0, 3.0, 0.0],
        ]
    )
    expected_shots_matrix = np.array(
        [
            [0.0, 5.0, 4.0],
            [5.0, 0.0, 6.0],
            [4.0, 6.0, 0.0],
        ]
    )

    np.testing.assert_allclose(store.counts_matrix(), expected_counts_matrix)
    np.testing.assert_allclose(store.shots_matrix(), expected_shots_matrix)


def test_variance_proxy_matches_bernoulli_formula() -> None:
    store = KernelStore(3)

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
    store.update(counts, shots)

    # p = 0.5 for all measured entries; p(1-p)/N = 0.25/N
    expected = np.array(
        [
            [0.0, 0.0625, 0.125],
            [0.0625, 0.0, 0.041666666666666664],
            [0.125, 0.041666666666666664, 0.0],
        ]
    )

    np.testing.assert_allclose(store.variance_proxy(), expected)


def test_reset_clears_all_state() -> None:
    store = KernelStore(3)

    counts = np.array(
        [
            [0, 1, 2],
            [0, 0, 3],
            [0, 0, 0],
        ],
        dtype=int,
    )
    shots = np.array(
        [
            [0, 4, 5],
            [0, 0, 6],
            [0, 0, 0],
        ],
        dtype=int,
    )
    store.update(counts, shots)
    store.reset()

    np.testing.assert_array_equal(store.counts_upper, np.zeros((3, 3), dtype=int))
    np.testing.assert_array_equal(store.shots_upper, np.zeros((3, 3), dtype=int))
    np.testing.assert_allclose(store.kernel_matrix(), np.eye(3))
    assert store.total_shots_spent() == 0
    assert store.num_measured_entries() == 0
