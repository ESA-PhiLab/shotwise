import numpy as np
import pytest

from shotwise.backends.qiskit_probability_backend import QiskitBackend

# ============================================================
# Dummy kernel (no Qiskit dependency)
# ============================================================


class DummyKernel:
    def evaluate(self, X1, X2):
        # deterministic kernel: dot product clipped to [0, 1]
        vals = np.sum(X1 * X2, axis=1)
        return np.clip(vals, 0.0, 1.0)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def simple_data():
    return np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ]
    )


@pytest.fixture
def backend(simple_data):
    return QiskitBackend(
        simple_data,
        kernel=DummyKernel(),
        shots=100,
        max_circuits_per_job=2,  # force batching
        rng=np.random.default_rng(0),
    )


# ============================================================
# Basic functionality
# ============================================================


def test_returns_correct_shape(backend):
    n = backend.n
    shot_matrix = np.ones((n, n), dtype=int) * 10

    counts = backend.sample_upper_triangle_counts(shot_matrix)

    assert counts.shape == (n, n)


def test_only_upper_triangle_filled(backend):
    n = backend.n
    shot_matrix = np.ones((n, n), dtype=int) * 10

    counts = backend.sample_upper_triangle_counts(shot_matrix)

    lower_indices = np.tril_indices(n)
    assert np.all(counts[lower_indices] == 0)


# ============================================================
# Zero-shot handling
# ============================================================


def test_zero_shots_produce_zero_counts(backend):
    shot_matrix = np.zeros((backend.n, backend.n), dtype=int)

    counts = backend.sample_upper_triangle_counts(shot_matrix)

    assert np.all(counts == 0)


# ============================================================
# Counts bounds
# ============================================================


def test_counts_within_shot_limits(backend):
    n = backend.n
    shot_matrix = np.ones((n, n), dtype=int) * 50

    counts = backend.sample_upper_triangle_counts(shot_matrix)

    rows, cols = np.triu_indices(n, k=1)

    for i, j in zip(rows, cols, strict=True):
        assert 0 <= counts[i, j] <= 50


# ============================================================
# Batching logic
# ============================================================


def test_batching_splits_pairs(backend):
    n = backend.n

    pairs = [(i, j, 10) for i in range(n) for j in range(i + 1, n)]

    batches = backend._split_into_batches(pairs)

    assert len(batches) > 1

    for batch in batches:
        assert len(batch) <= backend.max_circuits_per_job


# ============================================================
# Kernel effect
# ============================================================


def test_kernel_affects_counts(simple_data):
    backend = QiskitBackend(
        simple_data,
        kernel=DummyKernel(),
        shots=100,
        rng=np.random.default_rng(0),
    )

    shot_matrix = np.ones((3, 3), dtype=int) * 20

    counts = backend.sample_upper_triangle_counts(shot_matrix)

    assert np.any(counts > 0)


# ============================================================
# Reproducibility
# ============================================================


def test_rng_reproducibility(simple_data):
    kernel = DummyKernel()

    backend1 = QiskitBackend(
        simple_data,
        kernel=kernel,
        shots=50,
        rng=np.random.default_rng(42),
    )

    backend2 = QiskitBackend(
        simple_data,
        kernel=kernel,
        shots=50,
        rng=np.random.default_rng(42),
    )

    shot_matrix = np.ones((3, 3), dtype=int) * 10

    counts1 = backend1.sample_upper_triangle_counts(shot_matrix)
    counts2 = backend2.sample_upper_triangle_counts(shot_matrix)

    assert np.array_equal(counts1, counts2)


# ============================================================
# Edge cases
# ============================================================


def test_empty_pairs_returns_zero(backend):
    shot_matrix = np.zeros((backend.n, backend.n), dtype=int)

    counts = backend.sample_upper_triangle_counts(shot_matrix)

    assert np.all(counts == 0)


def test_invalid_kernel_rejected(simple_data):
    class BadKernel:
        pass

    with pytest.raises(ValueError):
        QiskitBackend(simple_data, kernel=BadKernel())
