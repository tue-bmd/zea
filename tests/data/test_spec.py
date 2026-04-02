import numpy as np
import pytest

from zea.data.spec import Dataset, Scan, Segmentation


def test_segmentation_spec():
    # Correct usage
    pixels = np.zeros((10, 256, 256, 1), dtype=np.uint8)
    labels = np.array(["background", "label1", "label2", "label3"], dtype=np.str_)
    extent = np.array([0.0, 1.0, 0.0, 1.0, -1.0, 0.0], dtype=np.float32)
    segmentation = Segmentation(pixels=pixels, labels=labels, extent=extent)
    assert segmentation.pixels.shape == (10, 256, 256, 1)
    assert segmentation.labels.shape == (4,)
    assert segmentation.extent.shape == (6,)

    # Incorrect usage: pixel values do not correspond to labels
    pixels_invalid = np.array([[[[0], [1]], [[2], [3]]], [[[4], [5]], [[6], [7]]]], dtype=np.uint8)
    with pytest.raises(
        ValueError, match="Segmentation pixels contain values that do not correspond to any label"
    ):
        Segmentation(pixels=pixels_invalid, labels=labels, extent=extent)


def _scan_minimal(n_frames: int = 3, n_tx: int = 2, n_el: int = 4, n_ax: int = 8, n_ch: int = 1):
    return {
        "n_ax": n_ax,
        "n_el": n_el,
        "n_tx": n_tx,
        "n_ch": n_ch,
        "n_frames": n_frames,
        "probe_geometry": np.zeros((n_el, 3), dtype=np.float32),
        "sampling_frequency": np.float32(30e6),
        "center_frequency": np.linspace(5e6, 6e6, n_tx, dtype=np.float32),
        "demodulation_frequency": np.linspace(5e6, 6e6, n_tx, dtype=np.float32),
        "initial_times": np.zeros((n_tx,), dtype=np.float32),
        "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
        "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
        "focus_distances": np.zeros((n_tx,), dtype=np.float32),
        "transmit_origins": np.zeros((n_tx, 3), dtype=np.float32),
        "polar_angles": np.zeros((n_tx,), dtype=np.float32),
        "azimuth_angles": np.zeros((n_tx,), dtype=np.float32),
        "time_to_next_transmit": np.ones((n_frames, n_tx), dtype=np.float32),
    }


def test_dataset_full_new_spec_from_dicts():
    n_frames, n_tx, n_el, n_ax, n_ch = 3, 2, 4, 8, 1

    dataset = Dataset(
        data={
            "raw_data": np.zeros((n_frames, n_tx, n_el, n_ax, n_ch), dtype=np.float32),
            "image": {
                "pixels": np.zeros((n_frames, 16, 12, 1), dtype=np.uint8),
                "extent": np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32),
            },
            "segmentation": {
                "pixels": np.zeros((n_frames, 16, 12, 1), dtype=np.uint8),
                "labels": np.array(["background", "tissue"], dtype=np.str_),
                "extent": np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32),
            },
            "sos_map": {
                "pixels": np.full((n_frames, 16, 12, 1), 1540.0, dtype=np.float32),
                "extent": np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32),
            },
            "strain": {
                "pixels": np.zeros((n_frames, 16, 12, 1), dtype=np.float32),
                "extent": np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32),
            },
            "swe": {
                "pixels": np.zeros((n_frames, 16, 12, 1), dtype=np.float32),
                "extent": np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32),
            },
            "tissue_doppler": {
                "pixels": np.zeros((n_frames, 16, 12, 1), dtype=np.float32),
                "extent": np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32),
            },
        },
        scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el, n_ax=n_ax, n_ch=n_ch),
        metadata={
            "subject": {"type": "human", "age": np.uint8(42), "sex": "f", "fat": np.float32(17.5)},
            "credit": "example-lab",
            "probe_orientation": {
                "pose": np.zeros((25, 6), dtype=np.float32),
                "offset": np.float32(0.0),
                "sampling_frequency": np.float32(50.0),
            },
            "voice_narration": {
                "samples": np.zeros((100, 1), dtype=np.uint8),
                "offset": np.float32(0.0),
                "sampling_frequency": np.float32(8000.0),
            },
            "ecg": {
                "samples": np.zeros((100, 1), dtype=np.uint8),
                "offset": np.float32(0.0),
                "sampling_frequency": np.float32(250.0),
            },
            "text_report": "normal acquisition",
            "annotations": {
                "anatomy": "heart",
                "view": np.array(["plax", "plax", "psax"], dtype=np.str_),
                "label": np.array(["normal", "normal", "normal"], dtype=np.str_),
                "image_quality": "high",
            },
        },
        metrics={
            "common_midpoint_phase_error": np.zeros((n_frames,), dtype=np.float32),
            "coherence_factor": np.ones((n_frames,), dtype=np.float32),
        },
    )

    assert dataset.data.raw_data.shape == (n_frames, n_tx, n_el, n_ax, n_ch)
    assert dataset.scan.t0_delays.shape == (n_tx, n_el)
    assert dataset.metadata.annotations.view.shape == (n_frames,)
    assert dataset.metrics.coherence_factor.shape == (n_frames,)


def test_scan_requires_required_fields():
    scan = _scan_minimal()
    scan.pop("n_tx")

    with pytest.raises(ValueError, match="Missing required field 'n_tx'"):
        Scan(**scan)


def test_scan_dimension_count_consistency():
    scan = _scan_minimal(n_tx=2)
    scan["initial_times"] = np.zeros((3,), dtype=np.float32)

    with pytest.raises(ValueError, match="Dimension 'n_tx' has inconsistent sizes"):
        Scan(**scan)


def test_optional_fields_can_be_omitted():
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

    dataset = Dataset(
        data={"raw_data": np.zeros((n_frames, n_tx, n_el, n_ax, n_ch), dtype=np.float32)},
        scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el, n_ax=n_ax, n_ch=n_ch),
        metadata={},
        metrics={},
    )

    assert dataset.metadata.subject is None
    assert dataset.metrics.common_midpoint_phase_error is None
