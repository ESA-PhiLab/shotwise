"""Tests for shotwise.algorithms.supportshot."""

from __future__ import annotations

import numpy as np
import pytest

from shotwise.algorithms.supportshot import SupportShot, SupportShotRoundRecord
from shotwise.backends.synthetic import SyntheticBernoulliBackend


@pytest.fixture
def kernel_matrix() -> np.ndarray:
    """Small normalized kernel with clear class structure."""
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


@pytest.fixture
def backend(kernel_matrix: np.ndarray) -> SyntheticBernoulliBackend:
    return SyntheticBernoulliBackend(kernel_matrix, random_state=0)


def test_fit_pilot_only_initializes_state(
    backend: SyntheticBernoulliBackend,
    labels: np.ndarray,
) -> None:
    model = SupportShot(
        total_budget=60,
        pilot_shots_per_entry=10,
        rounds=0,
        allocation_method="deterministic",
        random_state=0,
    )

    fitted = model.fit(labels, backend)

    assert fitted is model
    assert model.store_ is not None
    assert model.state_ is not None
    assert model.kernel_ is not None
    assert model.kernel_.shape == (4, 4)
    assert np.allclose(model.kernel_, model.kernel_.T)
    assert np.allclose(np.diag(model.kernel_), 1.0)
    assert model.remaining_budget_ == 0
    assert model.stopped_early_ is False

    assert len(model.history_) == 1
    record = model.history_[0]
    assert isinstance(record, SupportShotRoundRecord)
    assert record.stage == "pilot"
    assert record.round_index == 0
    assert record.budget == 60
    assert record.total_shots_spent == 60
    assert record.score_sum is None
    assert record.dual_change is None
    assert record.stopped_early is False
    assert np.allclose(record.kernel_matrix, model.kernel_)


def test_fit_adaptive_deterministic_consumes_total_budget(
    backend: SyntheticBernoulliBackend,
    labels: np.ndarray,
) -> None:
    model = SupportShot(
        total_budget=180,
        pilot_shots_per_entry=20,
        rounds=2,
        allocation_method="deterministic",
        random_state=0,
    )

    model.fit(labels, backend)

    assert model.store_ is not None
    assert model.state_ is not None
    assert model.kernel_ is not None
    assert np.allclose(model.kernel_, model.kernel_.T)
    assert np.allclose(np.diag(model.kernel_), 1.0)
    assert model.store_.total_shots_spent() == 180
    assert model.remaining_budget_ == 0

    # One pilot round plus two adaptive rounds.
    assert len(model.history_) == 3
    assert len(model.score_history_) == 2
    assert model.history_[0].stage == "pilot"
    assert all(record.stage == "adaptive" for record in model.history_[1:])
    assert all(record.budget == 30 for record in model.history_[1:])
    assert all(record.score_sum is not None for record in model.history_[1:])
    assert all(record.num_support >= 1 for record in model.history_)
    assert all(record.total_shots_spent <= 180 for record in model.history_)


def test_fit_with_large_stopping_epsilon_stops_early(
    backend: SyntheticBernoulliBackend,
    labels: np.ndarray,
) -> None:
    model = SupportShot(
        total_budget=210,
        pilot_shots_per_entry=20,
        rounds=3,
        stopping_epsilon=1e9,
        allocation_method="deterministic",
        random_state=0,
    )

    model.fit(labels, backend)

    assert model.store_ is not None
    assert model.stopped_early_ is True
    assert len(model.history_) == 2  # pilot + first adaptive round
    assert len(model.score_history_) == 1
    assert model.history_[-1].stopped_early is True
    assert model.history_[-1].dual_change is not None
    assert model.remaining_budget_ is not None
    assert model.remaining_budget_ > 0
    assert model.store_.total_shots_spent() < 210


def test_fit_rejects_pilot_budget_exceeding_total_budget(
    backend: SyntheticBernoulliBackend,
    labels: np.ndarray,
) -> None:
    model = SupportShot(
        total_budget=50,
        pilot_shots_per_entry=10,
        rounds=1,
    )

    with pytest.raises(ValueError, match="Pilot budget exceeds"):
        model.fit(labels, backend)


def test_fit_rejects_label_shape_mismatch(
    backend: SyntheticBernoulliBackend,
) -> None:
    model = SupportShot(
        total_budget=120,
        pilot_shots_per_entry=20,
        rounds=1,
    )

    with pytest.raises(ValueError, match=r"y must have shape \(4,\)"):
        model.fit(np.array([1, -1, 1]), backend)


def test_fit_rejects_backend_without_sampling_method(labels: np.ndarray) -> None:
    class BadBackend:
        n = 4

    model = SupportShot(
        total_budget=120,
        pilot_shots_per_entry=20,
        rounds=1,
    )

    with pytest.raises(ValueError, match="sample_upper_triangle_counts"):
        model.fit(labels, BadBackend())


def test_fit_rejects_backend_returning_wrong_shape(labels: np.ndarray) -> None:
    class WrongShapeBackend:
        n = 4

        def sample_upper_triangle_counts(self, shot_matrix: np.ndarray) -> np.ndarray:
            return np.zeros((3, 3), dtype=int)

    model = SupportShot(
        total_budget=120,
        pilot_shots_per_entry=20,
        rounds=1,
    )

    with pytest.raises(ValueError, match="incompatible shape"):
        model.fit(labels, WrongShapeBackend())


@pytest.mark.parametrize(
    ("kwargs", "message_part"),
    [
        ({"total_budget": -1}, "non-negative"),
        ({"pilot_shots_per_entry": -1}, "non-negative"),
        ({"rounds": -1}, "non-negative"),
        ({"lambda_weight": 1.5}, r"lie in \[0, 1\]"),
        ({"stopping_epsilon": -1e-3}, "non-negative"),
        ({"allocation_method": "unknown"}, "allocation_method must be"),
    ],
)
def test_constructor_rejects_invalid_configuration(
    kwargs: dict[str, object],
    message_part: str,
) -> None:
    base_kwargs = dict(total_budget=100)
    base_kwargs.update(kwargs)
    with pytest.raises(ValueError, match=message_part):
        SupportShot(**base_kwargs)


def test_early_stopping_sets_flag(backend, labels):
    from shotwise.allocation.stopping import DualChangeStopping

    model = SupportShot(
        total_budget=200,
        pilot_shots_per_entry=20,
        rounds=5,
        stopping=DualChangeStopping(epsilon=1e9),
        random_state=0,
    )

    model.fit(labels, backend)

    assert model.stopped_early_
