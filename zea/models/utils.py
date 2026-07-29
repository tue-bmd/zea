"""Utilities for models"""

import keras


def onnx2tf_saved_model_kwargs():  # pragma: no cover
    """Extra ``onnx2tf.convert`` arguments needed to still get a TensorFlow export.

    onnx2tf 2.6 made ``flatbuffer_direct`` the default backend, and that one only
    emits TFLite: the conversion helpers in this package ask for a SavedModel or a
    Keras v3 file and silently get neither. Selecting the classic exporter restores
    the pre-2.6 behaviour. Older onnx2tf versions have no such option and always
    write one, so the argument is only passed when it is understood.

    Returns:
        dict: Keyword arguments to splat into :func:`onnx2tf.convert`.
    """
    import inspect

    # The public convert() of recent versions is a ``**kwargs`` passthrough, so the
    # implementation is what has to be introspected. Both re-export the same object.
    from onnx2tf.onnx2tf import convert

    if "tflite_backend" in inspect.signature(convert).parameters:
        return {"tflite_backend": "tf_converter"}
    return {}


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
