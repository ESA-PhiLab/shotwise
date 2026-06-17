import numpy as np

from shotwise.algorithms.supportshot import SupportShotRoundRecord
from shotwise.storage.history import History


def make_dummy_record(round_index: int):
    n = 4
    return SupportShotRoundRecord(
        stage="adaptive",
        round_index=round_index,
        budget=10,
        allocation_matrix=np.ones((n, n)),
        observed_counts=np.zeros((n, n)),
        kernel_matrix=np.eye(n),
        total_shots_spent=10 * (round_index + 1),
        score_sum=1.0,
        num_support=2,
        dual_change=0.1 * round_index,
        stopped_early=False,
    )


# ============================================================
# Basic behavior
# ============================================================


def test_history_append_and_length():
    history = History()

    history.append(make_dummy_record(0))
    history.append(make_dummy_record(1))

    assert len(history) == 2


def test_history_indexing():
    history = History()

    r0 = make_dummy_record(0)
    history.append(r0)

    assert history[0].round_index == 0


def test_history_iteration_order():
    history = History()

    history.append(make_dummy_record(0))
    history.append(make_dummy_record(1))
    history.append(make_dummy_record(2))

    rounds = [r.round_index for r in history]
    assert rounds == [0, 1, 2]


# ============================================================
# Content integrity
# ============================================================


def test_history_stores_fields_correctly():
    history = History()

    rec = make_dummy_record(3)
    history.append(rec)

    stored = history[0]

    assert stored.round_index == 3
    assert np.isclose(stored.dual_change, 0.3)
    assert stored.num_support == 2


# ============================================================
# Memory safety (important!)
# ============================================================


def test_history_does_not_alias_kernel_matrix():
    history = History()

    rec = make_dummy_record(0)
    history.append(rec)

    rec.kernel_matrix[0, 0] = 999  # mutate original

    # Should not propagate if copy used correctly upstream
    stored = history[0]

    # NOTE: This test assumes copy is done before storing
    assert stored.kernel_matrix[0, 0] != 999
