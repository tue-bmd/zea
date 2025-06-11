"""Utilities for models"""

import keras


class LossTrackerWrapper:
    """A wrapper for Keras Mean metrics to track multiple loss values."""

    def __init__(self, prefix):
        """
        Initialize the loss tracker wrapper.

        Args:
            prefix (str): Prefix to use for the loss name. For example "n_loss" or "i_loss".
        """
        self.prefix = prefix
        self.trackers = {}

    def update_state(self, loss_value):
        """
        Update the tracker(s) with a loss value.

        If loss_value is a dict, then for each key a separate tracker is
        created (if not already created) and updated. The tracker's name will
        be <prefix>_<key>. If loss_value is not a dict, then a default tracker
        with name <prefix> is updated.

        Args:
            loss_value: A tensor or a dictionary mapping field names to tensors.
        """
        if isinstance(loss_value, dict):
            for key, value in loss_value.items():
                tracker_name = f"{self.prefix}_{key}"
                if tracker_name not in self.trackers:
                    self.trackers[tracker_name] = keras.metrics.Mean(name=tracker_name)
                self.trackers[tracker_name].update_state(value)
        else:
            if self.prefix not in self.trackers:
                self.trackers[self.prefix] = keras.metrics.Mean(name=self.prefix)
            self.trackers[self.prefix].update_state(loss_value)

    def result(self):
        """
        Return a dictionary with the current average results.
        """
        results = {}
        for _, tracker in self.trackers.items():
            # Use the tracker's name (e.g. "n_loss_a") if available
            results[tracker.name] = tracker.result()
        return results

    def reset_state(self):
        """
        Reset all the internal trackers.
        """
        for tracker in self.trackers.values():
            tracker.reset_state()

    def __iter__(self):
        return iter(self.trackers.values())
