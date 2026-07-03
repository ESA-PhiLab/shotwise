"""Tests for shotwise.backends.qiskit_shot_backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from shotwise.backends.qiskit_shot_backend import QiskitShotBackend


class FakeJoinedData:
    """Minimal object mimicking joined SamplerV2 data."""

    def __init__(self, counts: dict[str, int], num_bits: int = 2) -> None:
        self._counts = counts
        self.num_bits = num_bits

    def get_counts(self) -> dict[str, int]:
        return self._counts


class FakePubResult:
    """Minimal object mimicking a SamplerV2 PUB result."""

    def __init__(self, counts: dict[str, int], num_bits: int = 2) -> None:
        self._joined_data = FakeJoinedData(counts, num_bits=num_bits)

    def join_data(self) -> FakeJoinedData:
        return self._joined_data


class FakeJob:
    """Minimal Sampler job object."""

    def __init__(
        self,
        *,
        job_id: str,
        results: list[FakePubResult],
    ) -> None:
        self._job_id = job_id
        self._results = results

    def job_id(self) -> str:
        return self._job_id

    def result(self) -> list[FakePubResult]:
        return self._results


@dataclass
class SamplerCall:
    """Recorded sampler call."""

    pubs: list[Any]
    shots: int | None


class FakeSampler:
    """Fake SamplerV2-like primitive.

    Parameters
    ----------
    results_per_call
        A list where each element is the list of count dictionaries returned
        for one sampler.run(...) call.
    """

    def __init__(self, results_per_call: list[list[dict[str, int]]]) -> None:
        self.results_per_call = results_per_call
        self.calls: list[SamplerCall] = []

    def run(self, pubs: list[Any], *, shots: int | None = None) -> FakeJob:
        self.calls.append(SamplerCall(pubs=list(pubs), shots=shots))

        call_index = len(self.calls) - 1
        try:
            count_dicts = self.results_per_call[call_index]
        except IndexError as exc:
            raise AssertionError("FakeSampler received too many calls.") from exc

        results = [FakePubResult(counts, num_bits=2) for counts in count_dicts]
        return FakeJob(job_id=f"job-{call_index}", results=results)


def build_feature_map() -> tuple[QuantumCircuit, list[Any]]:
    """Build a tiny 2-qubit feature map and return ordered parameters."""
    theta = ParameterVector("x", 2)
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cz(0, 1)
    return qc, list(theta)


def build_dataset() -> np.ndarray:
    return np.array(
        [
            [0.10, 0.15],
            [0.55, 0.55],
            [1.20, 1.10],
        ],
        dtype=float,
    )


def test_pub_shots_backend_returns_direct_sampler_counts() -> None:
    """Requested shots should be sent to Sampler and counts returned directly."""
    x = build_dataset()
    feature_map, parameters = build_feature_map()

    sampler = FakeSampler(
        results_per_call=[
            [
                {"00": 3, "01": 4},
                {"00": 11, "11": 2},
            ]
        ]
    )

    backend = QiskitShotBackend(
        x,
        feature_map=feature_map,
        sampler=sampler,
        input_parameters=parameters,
        max_circuits_per_job=10,
        use_pub_shots=True,
    )

    shot_matrix = np.zeros((3, 3), dtype=int)
    shot_matrix[0, 1] = 7
    shot_matrix[1, 2] = 13

    counts = backend.sample_upper_triangle_counts(shot_matrix, round_id=5)

    expected = np.zeros((3, 3), dtype=int)
    expected[0, 1] = 3
    expected[1, 2] = 11
    np.testing.assert_array_equal(counts, expected)

    assert len(sampler.calls) == 1
    call = sampler.calls[0]
    assert call.shots is None
    assert len(call.pubs) == 2

    submitted_shots = [pub[2] for pub in call.pubs]
    assert submitted_shots == [7, 13]

    assert len(backend.job_history_) == 1
    job_record = backend.job_history_[0]
    assert job_record.round_id == 5
    assert job_record.n_circuits == 2
    assert job_record.total_shots == 20
    assert job_record.job_id == "job-0"


def test_backend_does_not_locally_resample_counts() -> None:
    """Returned counts must equal sampler all-zero counts exactly."""
    x = build_dataset()
    feature_map, parameters = build_feature_map()

    sampler = FakeSampler(
        results_per_call=[
            [
                {"00": 0, "01": 7},
            ]
        ]
    )

    backend = QiskitShotBackend(
        x,
        feature_map=feature_map,
        sampler=sampler,
        input_parameters=parameters,
        max_circuits_per_job=10,
        use_pub_shots=True,
    )

    shot_matrix = np.zeros((3, 3), dtype=int)
    shot_matrix[0, 1] = 7

    counts = backend.sample_upper_triangle_counts(shot_matrix)

    assert counts[0, 1] == 0
    assert counts.sum() == 0


def test_backend_batches_by_max_circuits_per_job() -> None:
    """max_circuits_per_job should split active pairs into multiple jobs."""
    x = build_dataset()
    feature_map, parameters = build_feature_map()

    sampler = FakeSampler(
        results_per_call=[
            [{"00": 2, "01": 3}],
            [{"00": 4, "10": 1}],
        ]
    )

    backend = QiskitShotBackend(
        x,
        feature_map=feature_map,
        sampler=sampler,
        input_parameters=parameters,
        max_circuits_per_job=1,
        use_pub_shots=True,
    )

    shot_matrix = np.zeros((3, 3), dtype=int)
    shot_matrix[0, 1] = 5
    shot_matrix[0, 2] = 5

    counts = backend.sample_upper_triangle_counts(shot_matrix, round_id=2)

    assert counts[0, 1] == 2
    assert counts[0, 2] == 4
    assert len(sampler.calls) == 2
    assert len(backend.job_history_) == 2
    assert [record.round_id for record in backend.job_history_] == [2, 2]


def test_grouped_by_shots_fallback_uses_run_level_shots() -> None:
    """Fallback mode should group circuits by identical shot count."""
    x = build_dataset()
    feature_map, parameters = build_feature_map()

    sampler = FakeSampler(
        results_per_call=[
            [
                {"00": 2, "01": 5},
                {"00": 4, "10": 3},
            ],
            [
                {"00": 9, "11": 4},
            ],
        ]
    )

    backend = QiskitShotBackend(
        x,
        feature_map=feature_map,
        sampler=sampler,
        input_parameters=parameters,
        max_circuits_per_job=10,
        use_pub_shots=False,
    )

    shot_matrix = np.zeros((3, 3), dtype=int)
    shot_matrix[0, 1] = 7
    shot_matrix[0, 2] = 7
    shot_matrix[1, 2] = 13

    counts = backend.sample_upper_triangle_counts(shot_matrix)

    assert counts[0, 1] == 2
    assert counts[0, 2] == 4
    assert counts[1, 2] == 9

    assert len(sampler.calls) == 2
    assert sampler.calls[0].shots == 7
    assert sampler.calls[1].shots == 13

    assert len(backend.job_history_) == 2
    assert backend.job_history_[0].total_shots == 14
    assert backend.job_history_[1].total_shots == 13


def test_inactive_pairs_return_zero_counts_without_sampler_call() -> None:
    """All-zero shot matrix should not submit any sampler job."""
    x = build_dataset()
    feature_map, parameters = build_feature_map()
    sampler = FakeSampler(results_per_call=[])

    backend = QiskitShotBackend(
        x,
        feature_map=feature_map,
        sampler=sampler,
        input_parameters=parameters,
    )

    shot_matrix = np.zeros((3, 3), dtype=int)
    counts = backend.sample_upper_triangle_counts(shot_matrix)

    np.testing.assert_array_equal(counts, np.zeros((3, 3), dtype=int))
    assert sampler.calls == []
    assert backend.job_history_ == []


def test_rejects_invalid_shot_matrix_shape() -> None:
    x = build_dataset()
    feature_map, parameters = build_feature_map()
    sampler = FakeSampler(results_per_call=[])

    backend = QiskitShotBackend(
        x,
        feature_map=feature_map,
        sampler=sampler,
        input_parameters=parameters,
    )

    with pytest.raises(ValueError, match="shot_matrix must have shape"):
        backend.sample_upper_triangle_counts(np.zeros((2, 2), dtype=int))


def test_rejects_invalid_input_parameter_count() -> None:
    x = build_dataset()
    feature_map, parameters = build_feature_map()
    sampler = FakeSampler(results_per_call=[])

    with pytest.raises(ValueError, match="Number of input parameters"):
        QiskitShotBackend(
            x,
            feature_map=feature_map,
            sampler=sampler,
            input_parameters=parameters[:1],
        )


def test_rejects_non_positive_max_circuits_per_job() -> None:
    x = build_dataset()
    feature_map, parameters = build_feature_map()
    sampler = FakeSampler(results_per_call=[])

    with pytest.raises(ValueError, match="max_circuits_per_job"):
        QiskitShotBackend(
            x,
            feature_map=feature_map,
            sampler=sampler,
            input_parameters=parameters,
            max_circuits_per_job=0,
        )


def test_raises_if_sampler_result_length_mismatches_submission() -> None:
    x = build_dataset()
    feature_map, parameters = build_feature_map()

    sampler = FakeSampler(results_per_call=[[{"00": 1}]])

    backend = QiskitShotBackend(
        x,
        feature_map=feature_map,
        sampler=sampler,
        input_parameters=parameters,
        max_circuits_per_job=10,
        use_pub_shots=True,
    )

    shot_matrix = np.zeros((3, 3), dtype=int)
    shot_matrix[0, 1] = 5
    shot_matrix[0, 2] = 5

    with pytest.raises(RuntimeError, match="result length does not match"):
        backend.sample_upper_triangle_counts(shot_matrix)
