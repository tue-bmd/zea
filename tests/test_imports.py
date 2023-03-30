"""Tries importing all modules in the package."""

def test_ui():
    import usbmd.ui

def test_scan():
    import usbmd.scan

def test_processing():
    import usbmd.processing

def test_probes():
    import usbmd.probes

def test_generate():
    import usbmd.generate

def test_datasets():
    import usbmd.datasets

def test_common():
    import usbmd.common

# Utils
def test_utils_config_validation():
    import usbmd.utils.config_validation

def test_utils_config():
    import usbmd.utils.config

def test_utils_convert():
    import usbmd.utils.convert

def test_utils_git_info():
    import usbmd.utils.git_info

def test_utils_metrics():
    import usbmd.utils.metrics

def test_utils_pixelgrid():
    import usbmd.utils.pixelgrid

def test_utils_read_h5():
    import usbmd.utils.read_h5

def test_utils_registration():
    import usbmd.utils.registration

def test_utils_selection_tool():
    import usbmd.utils.selection_tool

def test_utils_simulator():
    import usbmd.utils.simulator

def test_utils_utils():
    import usbmd.utils.utils

def test_utils_video():
    import usbmd.utils.video

# Pytorch ultrasound
def test_pytorch_ultrasound_processing():
    import usbmd.pytorch_ultrasound.processing

def test_pytorch_ultrasound_beamformers():
    import usbmd.pytorch_ultrasound.layers.beamformers

# Tensorflow ultrasound
def test_tensorflow_ultrasound_processing():
    import usbmd.tensorflow_ultrasound.processing

def test_tensorflow_ultrasound_dataloader():
    import usbmd.tensorflow_ultrasound.dataloader

def test_tensorflow_ultrasound_losses():
    import usbmd.tensorflow_ultrasound.losses

def test_tensorflow_ultrasound_beamformer():
    import usbmd.tensorflow_ultrasound.layers.beamformers