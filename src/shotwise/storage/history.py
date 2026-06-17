"""
History tracking for ShotWise algorithms.

Stores per-round state of the algorithm:
- kernel estimates
- allocations
- dual evolution
- dual stability

This is a *pure data container* (no evaluation logic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np

# ============================================================
# History entry
# ============================================================


@dataclass
class HistoryEntry:
    """
    Snapshot of the algorithm state at a given round.
    """

    # --- identification ---
    round_index: int
    stage: str  # "pilot" or "adaptive"

    # --- core state ---
    kernel_matrix: np.ndarray

    # --- allocation ---
    allocation: np.ndarray
    weights: Optional[np.ndarray] = None

    # --- convergence / dynamics ---
    dual_change: Optional[float] = None

    # --- bookkeeping ---
    budget: int = 0
    total_shots_spent: int = 0
    stopped_early: bool = False

    def copy(self) -> "HistoryEntry":
        """
        Return a deep copy of the entry (safe for downstream use).
        """
        return HistoryEntry(
            round_index=self.round_index,
            stage=self.stage,
            kernel_matrix=self.kernel_matrix.copy(),
            allocation=self.allocation.copy(),
            weights=None if self.weights is None else self.weights.copy(),
            dual_change=self.dual_change,
            budget=self.budget,
            total_shots_spent=self.total_shots_spent,
            stopped_early=self.stopped_early,
        )


# ============================================================
# History container
# ============================================================


class History:
    """
    Container for HistoryEntry objects.

    Behaves like a list but with semantic meaning.
    """

    def __init__(self) -> None:
        self._entries: List[HistoryEntry] = []

    # -----------------------------------
    # core API
    # -----------------------------------

    def append(self, entry: HistoryEntry) -> None:
        """
        Append a *safe copy* of the entry.

        This enforces immutability of stored history even if the caller
        later mutates the original object.
        """
        self._entries.append(self._copy_entry(entry))

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> HistoryEntry:
        return self._entries[idx]

    def __iter__(self) -> Iterator[HistoryEntry]:
        return iter(self._entries)

    # -----------------------------------
    # internal helpers
    # -----------------------------------

    def _copy_entry(self, entry):
        """
        Create a defensive copy of an entry.

        Works generically for dataclasses like SupportShotRoundRecord.
        Copies any numpy arrays it finds.
        """
        data = {}

        for key, value in entry.__dict__.items():
            if isinstance(value, np.ndarray):
                data[key] = value.copy()
            else:
                data[key] = value

        return type(entry)(**data)

    # -----------------------------------
    # convenience
    # -----------------------------------

    @property
    def rounds(self) -> List[int]:
        return [e.round_index for e in self._entries]

    @property
    def stages(self) -> List[str]:
        return [e.stage for e in self._entries]

    @property
    def dual_changes(self) -> List[Optional[float]]:
        return [e.dual_change for e in self._entries]

    @property
    def kernels(self) -> List[np.ndarray]:
        return [e.kernel_matrix for e in self._entries]

    def last(self) -> HistoryEntry:
        if not self._entries:
            raise RuntimeError("History is empty")
        return self._entries[-1]

    def to_list(self) -> List[HistoryEntry]:
        """
        Return deep copies of all entries (safe for external use).
        """
        return [e.copy() for e in self._entries]
