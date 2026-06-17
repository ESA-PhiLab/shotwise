"""
Stopping policies for adaptive allocation.

Provides reusable criteria for terminating iterative algorithms
based on convergence signals (e.g. dual stability).
"""

from __future__ import annotations

from dataclasses import dataclass

# ============================================================
# Base interface
# ============================================================


class StoppingCriterion:
    """Abstract interface for stopping rules."""

    def should_stop(self, *, round_index: int, dual_change: float | None) -> bool:
        raise NotImplementedError


# ============================================================
# Dual-change stopping
# ============================================================


@dataclass
class DualChangeStopping(StoppingCriterion):
    """
    Stop when the relative dual change is below a threshold.

    Parameters
    ----------
    epsilon
        Threshold for convergence.
    patience
        Optional number of consecutive rounds satisfying the criterion.
    min_rounds
        Minimum number of rounds before stopping is allowed.
    """

    epsilon: float
    patience: int = 1
    min_rounds: int = 1

    def __post_init__(self) -> None:
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.min_rounds < 0:
            raise ValueError("min_rounds must be >= 0")

        self._counter = 0

    def should_stop(
        self,
        *,
        round_index: int,
        dual_change: float | None,
    ) -> bool:
        """Return True if stopping condition is met."""
        # Not enough rounds yet
        if round_index < self.min_rounds:
            self._counter = 0
            return False

        if dual_change is None:
            return False

        if dual_change < self.epsilon:
            self._counter += 1
        else:
            self._counter = 0

        return self._counter >= self.patience
