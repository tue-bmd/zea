"""Tests for :mod:`zea.models.utils`."""

import numpy as np

from zea.models.utils import LossTrackerWrapper


def test_tracks_a_single_loss_under_the_prefix():
    """A plain loss value lands in one tracker named after the prefix."""
    tracker = LossTrackerWrapper("n_loss")

    tracker.update_state(2.0)
    tracker.update_state(4.0)

    result = tracker.result()
    assert list(result) == ["n_loss"]
    np.testing.assert_allclose(float(result["n_loss"]), 3.0)  # running mean


def test_tracks_one_metric_per_key_of_a_dict_loss():
    """A dict of losses gets a ``<prefix>_<key>`` tracker per entry, created on demand."""
    tracker = LossTrackerWrapper("i_loss")

    tracker.update_state({"a": 1.0, "b": 10.0})
    tracker.update_state({"a": 3.0, "b": 20.0})

    result = tracker.result()
    assert sorted(result) == ["i_loss_a", "i_loss_b"]
    np.testing.assert_allclose(float(result["i_loss_a"]), 2.0)
    np.testing.assert_allclose(float(result["i_loss_b"]), 15.0)


def test_reset_state_clears_every_tracker():
    """Resetting between epochs zeroes all the running means."""
    tracker = LossTrackerWrapper("loss")
    tracker.update_state({"a": 5.0, "b": 7.0})

    tracker.reset_state()

    assert all(float(value) == 0.0 for value in tracker.result().values())


def test_iterating_yields_the_keras_metrics():
    """Iteration exposes the underlying metrics, so Keras can pick them up."""
    tracker = LossTrackerWrapper("loss")
    tracker.update_state({"a": 1.0, "b": 2.0})

    metrics = list(tracker)

    assert sorted(metric.name for metric in metrics) == ["loss_a", "loss_b"]
