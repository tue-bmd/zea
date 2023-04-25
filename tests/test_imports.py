"""Tries importing all modules in the package."""
# pylint: disable=unused-import, import-outside-toplevel

def test_ui():
    """Tests importing the usbmd.ui module."""
    import usbmd.ui

def test_scan():
    """Tests importing the usbmd.scan module."""
    import usbmd.scan

def test_processing():
    """Tests importing the usbmd.processing module."""
    import usbmd.processing

def test_probes():
    """Tests importing the usbmd.probes module."""
    import usbmd.probes

def test_generate():
    """Tests importing the usbmd.generate module."""
    import usbmd.generate

def test_datasets():
    """Tests importing the usbmd.datasets module."""
    import usbmd.datasets

def test_common():
    """Tests importing the usbmd.common module."""
    import usbmd.common

# Utils
def test_utils_config_validation():
    """Tests importing the usbmd.utils.config_validation module."""
    import usbmd.utils.config_validation

def test_utils_config():
    """Tests importing the usbmd.utils.config module."""
    import usbmd.utils.config

def test_utils_convert():
    """Tests importing the usbmd.utils.convert module."""
    import usbmd.utils.convert

def test_utils_git_info():
    """Tests importing the usbmd.utils.git_info module."""
    import usbmd.utils.git_info

def test_utils_metrics():
    """Tests importing the usbmd.utils.metrics module."""
    import usbmd.utils.metrics

def test_utils_pixelgrid():
    """Tests importing the usbmd.utils.pixelgrid module."""
    import usbmd.utils.pixelgrid

def test_utils_read_h5():
    """Tests importing the usbmd.utils.read_h5 module."""
    import usbmd.utils.read_h5

def test_utils_registration():
    """Tests importing the usbmd.utils.registration module."""
    import usbmd.utils.registration

def test_utils_selection_tool():
    """Tests importing the usbmd.utils.selection_tool module."""
    import usbmd.utils.selection_tool

def test_utils_simulator():
    """Tests importing the usbmd.utils.simulator module."""
    import usbmd.utils.simulator

def test_utils_utils():
    """Tests importing the usbmd.utils.utils module."""
    import usbmd.utils.utils

def test_utils_video():
    """Tests importing the usbmd.utils.video module."""
    import usbmd.utils.video

# Pytorch ultrasound
def test_pytorch_ultrasound_processing():
    """Tests importing the usbmd.pytorch_ultrasound.processing module."""
    import usbmd.pytorch_ultrasound.processing

def test_pytorch_ultrasound_beamformers():
    """Tests importing the usbmd.pytorch_ultrasound.beamformers module."""
    import usbmd.pytorch_ultrasound.layers.beamformers

# Tensorflow ultrasound
def test_tensorflow_ultrasound_processing():
    """Tests importing the usbmd.tensorflow_ultrasound.processing module."""
    import usbmd.tensorflow_ultrasound.processing

def test_tensorflow_ultrasound_dataloader():
    """Tests importing the usbmd.tensorflow_ultrasound.dataloader module."""
    import usbmd.tensorflow_ultrasound.dataloader

def test_tensorflow_ultrasound_losses():
    """Tests importing the usbmd.tensorflow_ultrasound.losses module."""
    import usbmd.tensorflow_ultrasound.losses

def test_tensorflow_ultrasound_beamformer():
    """Tests importing the usbmd.tensorflow_ultrasound.beamformers module."""
    import usbmd.tensorflow_ultrasound.layers.beamformers
