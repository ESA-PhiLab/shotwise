"""SupportShot adaptive allocation algorithm for ShotWise.

This module implements a clean, modular proof-of-concept of the adaptive
shot-allocation strategy described in the paper. The algorithm is currently
designed around a backend that can sample upper-triangular Bernoulli counts,
such as :class:`shotwise.backends.synthetic.SyntheticBernoulliBackend`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shotwise.allocation.policy import (
    AllocationResult,
    allocate_deterministic_rounding,
    allocate_multinomial,
)
from shotwise.allocation.scores import ScoreComponents, compute_supportshot_scores
from shotwise.allocation.stopping import DualChangeStopping
from shotwise.learning.svm import SVMState, fit_precomputed_svm_state
from shotwise.storage.history import History
from shotwise.storage.kernel_store import KernelStore
from shotwise.utils.matrix import num_upper_triangle_entries, upper_triangle_only


@dataclass(frozen=True)
class SupportShotRoundRecord:
    """Summary of one pilot or adaptive round.

    Parameters
    ----------
    stage
        Either ``"pilot"`` or ``"adaptive"``.
    round_index
        Zero-based round index. The pilot is always index 0.
    budget
        Number of shots allocated during this round.
    allocation_matrix
        Strict upper-triangular integer matrix of requested shots.
    observed_counts
        Strict upper-triangular integer matrix of observed Bernoulli
        successes returned by the backend.
    kernel_matrix
        Full symmetric kernel estimate after the round update.
    total_shots_spent
        Cumulative number of shots consumed after the update.
    score_sum
        Sum of the adaptive score matrix used for this round. ``None`` for
        the pilot phase.
    num_support
        Number of support vectors after fitting the SVM on the updated
        kernel estimate.
    dual_change
        Relative change in the signed dual vector compared with the previous
        round. ``None`` for the pilot phase.
    stopped_early
        Whether this round triggered the early-stopping criterion.
    """

    stage: str
    round_index: int
    budget: int
    allocation_matrix: np.ndarray
    observed_counts: np.ndarray
    kernel_matrix: np.ndarray
    total_shots_spent: int
    score_sum: float | None
    num_support: int
    dual_change: float | None
    stopped_early: bool


class SupportShot:
    """Adaptive shot allocation for kernelized SVMs.

    Parameters
    ----------
    total_budget
        Total number of measurement shots available.
    pilot_shots_per_entry
        Uniform number of pilot shots assigned to each independent
        off-diagonal kernel entry.
    rounds
        Number of adaptive rounds after the pilot phase.
    lambda_weight
        Mixing weight between sensitivity and instability scores.
    svm_c
        SVM regularization parameter.
    stopping_epsilon
        Optional early-stopping threshold on the relative change of the
        signed dual coefficients.
    use_confidence_weighting
        Whether to divide dual coefficients by ``sigma_safe`` inside the
        sensitivity score.
    sigma_floor
        Stabilization constant used when forming ``sigma_safe``.
    hardware_floor
        Optional additive hardware variance floor.
    prior_success
        Beta-prior pseudocount for the smoothed probability numerator.
    prior_failure
        Beta-prior pseudocount for the smoothed probability denominator.
    epsilon
        Small positive constant added in the instability probability.
    allocation_method
        Either ``"multinomial"`` or ``"deterministic"``.
    random_state
        Optional seed or NumPy ``Generator`` used for stochastic allocation
        policies.
    """

    def __init__(
        self,
        *,
        total_budget: int,
        pilot_shots_per_entry: int = 20,
        rounds: int = 3,
        lambda_weight: float = 0.5,
        svm_c: float = 1.0,
        stopping: DualChangeStopping | None = None,
        stopping_epsilon: float | None = None,
        use_confidence_weighting: bool = False,
        sigma_floor: float = 1e-8,
        hardware_floor: float = 0.0,
        prior_success: float = 1.0,
        prior_failure: float = 1.0,
        epsilon: float = 1e-9,
        allocation_method: str = "multinomial",
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        if total_budget < 0:
            raise ValueError(f"total_budget must be non-negative, got {total_budget}.")
        if pilot_shots_per_entry < 0:
            raise ValueError(
                "pilot_shots_per_entry must be non-negative, "
                f"got {pilot_shots_per_entry}."
            )
        if rounds < 0:
            raise ValueError(f"rounds must be non-negative, got {rounds}.")
        if not 0.0 <= lambda_weight <= 1.0:
            raise ValueError(f"lambda_weight must lie in [0, 1], got {lambda_weight}.")
        if stopping_epsilon is not None and stopping_epsilon < 0.0:
            raise ValueError(
                "stopping_epsilon must be non-negative when provided, "
                f"got {stopping_epsilon}."
            )
        # -----------------------------------
        # Backward compatibility handling
        # -----------------------------------
        if stopping is not None and stopping_epsilon is not None:
            raise ValueError(
                "Provide either 'stopping' or 'stopping_epsilon', not both."
            )

        if stopping is None and stopping_epsilon is not None:
            # create stopping policy automatically
            stopping = DualChangeStopping(epsilon=stopping_epsilon)
        if allocation_method not in {"multinomial", "deterministic"}:
            raise ValueError(
                "allocation_method must be either 'multinomial' or "
                f"'deterministic', got {allocation_method!r}."
            )

        self.total_budget = int(total_budget)
        self.pilot_shots_per_entry = int(pilot_shots_per_entry)
        self.rounds = int(rounds)
        self.lambda_weight = float(lambda_weight)
        self.svm_c = float(svm_c)
        self.stopping = stopping
        self.stopping_epsilon = stopping_epsilon
        self.use_confidence_weighting = bool(use_confidence_weighting)
        self.sigma_floor = float(sigma_floor)
        self.hardware_floor = float(hardware_floor)
        self.prior_success = float(prior_success)
        self.prior_failure = float(prior_failure)
        self.epsilon = float(epsilon)
        self.allocation_method = allocation_method
        self._rng = _make_rng(random_state)

        self.store_: KernelStore | None = None
        self.state_: SVMState | None = None
        self.kernel_: np.ndarray | None = None
        self.history_: list[SupportShotRoundRecord] = []
        self.score_history_: list[ScoreComponents] = []
        self.stopped_early_: bool = False
        self.remaining_budget_: int | None = None

    def fit(self, y: np.ndarray, backend: object) -> SupportShot:
        """Run the SupportShot adaptive allocation procedure.

        Parameters
        ----------
        y
            Binary labels in ``{-1, +1}``.
        backend
            Measurement backend with attributes/methods compatible with the
            current proof-of-concept, namely ``n`` and
            ``sample_upper_triangle_counts(shot_matrix)``.

        Returns
        -------
        SupportShot
            The fitted estimator.
        """
        n = _infer_backend_dimension(backend)
        labels = np.asarray(y)
        if labels.shape != (n,):
            raise ValueError(f"y must have shape ({n},), got {labels.shape}.")

        pilot_budget = self.pilot_shots_per_entry * num_upper_triangle_entries(n)
        if pilot_budget > self.total_budget:
            raise ValueError(
                "Pilot budget exceeds the total budget: "
                f"pilot requires {pilot_budget}, total_budget={self.total_budget}."
            )

        self.store_ = KernelStore(n, diagonal_value=1.0)
        self.history_ = History()
        self.score_history_ = []
        self.stopped_early_ = False

        # Pilot phase
        pilot_allocation = upper_triangle_only(
            self.pilot_shots_per_entry * np.ones((n, n), dtype=int)
        )
        pilot_counts = _sample_backend_counts(backend, pilot_allocation)
        self.store_.update(pilot_counts, pilot_allocation)
        state = fit_precomputed_svm_state(
            self.store_.kernel_matrix(),
            labels,
            c=self.svm_c,
        )

        self.state_ = state
        self.kernel_ = self.store_.kernel_matrix()
        self.remaining_budget_ = self.total_budget - pilot_budget
        self.history_.append(
            SupportShotRoundRecord(
                stage="pilot",
                round_index=0,
                budget=pilot_budget,
                allocation_matrix=pilot_allocation.copy(),
                observed_counts=pilot_counts.copy(),
                kernel_matrix=self.kernel_.copy(),
                total_shots_spent=self.store_.total_shots_spent(),
                score_sum=None,
                num_support=state.num_support,
                dual_change=None,
                stopped_early=False,
            )
        )

        if self.rounds == 0 or self.remaining_budget_ == 0:
            return self

        previous_signed_dual = state.signed_dual.copy()
        round_budgets = _split_budget(self.remaining_budget_, self.rounds)

        for adaptive_round, budget in enumerate(round_budgets, start=1):
            if budget <= 0:
                continue

            components = compute_supportshot_scores(
                self.store_,
                state,
                lambda_weight=self.lambda_weight,
                use_confidence_weighting=self.use_confidence_weighting,
                sigma_floor=self.sigma_floor,
                hardware_floor=self.hardware_floor,
                prior_success=self.prior_success,
                prior_failure=self.prior_failure,
                epsilon=self.epsilon,
            )
            self.score_history_.append(components)
            score_sum = float(components.final_scores.sum())

            if score_sum <= 0.0:
                break

            allocation_result = self._allocate(components.final_scores, budget)
            observed_counts = _sample_backend_counts(
                backend,
                allocation_result.allocation_matrix,
            )
            self.store_.update(
                observed_counts,
                allocation_result.allocation_matrix,
            )

            state = fit_precomputed_svm_state(
                self.store_.kernel_matrix(),
                labels,
                c=self.svm_c,
            )
            self.state_ = state
            self.kernel_ = self.store_.kernel_matrix()

            rel_change = _relative_change(
                previous_signed_dual,
                state.signed_dual,
            )
            if self.stopping is not None:
                stopped_early = self.stopping.should_stop(
                    round_index=adaptive_round,
                    dual_change=rel_change,
                )
            else:
                stopped_early = False

            self.history_.append(
                SupportShotRoundRecord(
                    stage="adaptive",
                    round_index=adaptive_round,
                    budget=budget,
                    allocation_matrix=allocation_result.allocation_matrix.copy(),
                    observed_counts=observed_counts.copy(),
                    kernel_matrix=self.kernel_.copy(),
                    total_shots_spent=self.store_.total_shots_spent(),
                    score_sum=score_sum,
                    num_support=state.num_support,
                    dual_change=rel_change,
                    stopped_early=stopped_early,
                )
            )

            previous_signed_dual = state.signed_dual.copy()
            self.remaining_budget_ = self.total_budget - self.store_.total_shots_spent()

            if stopped_early:
                self.stopped_early_ = True
                break

        if self.remaining_budget_ is None:
            self.remaining_budget_ = self.total_budget - self.store_.total_shots_spent()
        return self

    def _allocate(self, scores: np.ndarray, budget: int) -> AllocationResult:
        """Dispatch to the selected allocation policy."""
        if self.allocation_method == "multinomial":
            return allocate_multinomial(
                scores,
                budget,
                random_state=self._rng,
            )
        return allocate_deterministic_rounding(scores, budget)


def _infer_backend_dimension(backend: object) -> int:
    """Infer the number of samples supported by the backend."""
    if hasattr(backend, "n"):
        n = int(backend.n)
        if n <= 0:
            raise ValueError(f"backend.n must be positive, got {n}.")
        return n

    if hasattr(backend, "kernel_matrix"):
        kernel = np.asarray(backend.kernel_matrix)
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError(
                "backend.kernel_matrix must be a square array if backend.n "
                "is not provided."
            )
        return int(kernel.shape[0])

    raise ValueError(
        "backend must expose either an integer attribute 'n' or a square "
        "array attribute 'kernel_matrix'."
    )


def _sample_backend_counts(backend: object, shot_matrix: np.ndarray) -> np.ndarray:
    """Request Bernoulli counts from a backend."""
    if not hasattr(backend, "sample_upper_triangle_counts"):
        raise ValueError(
            "backend must implement sample_upper_triangle_counts(shot_matrix)."
        )
    counts = np.asarray(backend.sample_upper_triangle_counts(shot_matrix), dtype=int)
    if counts.shape != shot_matrix.shape:
        raise ValueError(
            "backend returned counts with incompatible shape: "
            f"expected {shot_matrix.shape}, got {counts.shape}."
        )
    return counts


def _relative_change(previous: np.ndarray, current: np.ndarray) -> float:
    """Compute relative change between two vectors with a stable denominator."""
    diff = np.linalg.norm(current - previous)
    denom = np.linalg.norm(previous) + 1e-12
    return float(diff / denom)


def _split_budget(total_budget: int, rounds: int) -> list[int]:
    """Split a remaining budget as evenly as possible across rounds."""
    if rounds <= 0:
        return []
    base = total_budget // rounds
    remainder = total_budget % rounds
    budgets = [base] * rounds
    for i in range(remainder):
        budgets[i] += 1
    return budgets


def _make_rng(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    """Normalize a random-state input to a NumPy Generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)
