from shotwise.allocation.stopping import DualChangeStopping

# ============================================================
# Basic behavior
# ============================================================


def test_stopping_triggers_below_threshold():
    stop = DualChangeStopping(epsilon=0.1)

    should_stop = stop.should_stop(round_index=3, dual_change=0.05)

    assert should_stop


def test_stopping_not_triggered_above_threshold():
    stop = DualChangeStopping(epsilon=0.1)

    should_stop = stop.should_stop(round_index=3, dual_change=0.2)

    assert not should_stop


# ============================================================
# min_rounds behavior
# ============================================================


def test_stopping_respects_min_rounds():
    stop = DualChangeStopping(epsilon=0.1, min_rounds=5)

    # Below threshold but too early
    should_stop = stop.should_stop(round_index=3, dual_change=0.01)

    assert not should_stop


# ============================================================
# patience behavior
# ============================================================


def test_stopping_requires_patience():
    stop = DualChangeStopping(epsilon=0.1, patience=2)

    # first small change → not yet
    assert not stop.should_stop(round_index=1, dual_change=0.01)

    # second small change → triggers
    assert stop.should_stop(round_index=2, dual_change=0.01)


def test_stopping_resets_patience_on_spike():
    stop = DualChangeStopping(epsilon=0.1, patience=2)

    assert not stop.should_stop(round_index=1, dual_change=0.01)
    assert not stop.should_stop(round_index=2, dual_change=0.5)  # reset
    assert not stop.should_stop(round_index=3, dual_change=0.01)
    assert stop.should_stop(round_index=4, dual_change=0.01)


# ============================================================
# Edge cases
# ============================================================


def test_stopping_handles_none_dual_change():
    stop = DualChangeStopping(epsilon=0.1)

    should_stop = stop.should_stop(round_index=3, dual_change=None)

    assert not should_stop
