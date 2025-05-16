<!-- This is the readme for the github page (more complete readme for pdocs can be found in usmbd/README.md) -->
# usbmd <img src="docs/_static/usbmd_logo_v3.svg" style="float: right; width: 20%; height: 20%;" align="right" alt="usbmd Logo" />

The ultrasound toolbox (usbmd) is a collection of ultrasound tools (Python) such as beamforming code, visualization tools and deep learning scripts. Check out the full documentation [here](http://131.155.124.215:6001/) (only available within the TU/e network).

The idea of this toolbox is that it is self-sustained, meaning ultrasound researchers can use the tools to create new models / algorithms and after completed, can add them to the toolbox. This repository is being maintained by researchers from the [BM/d lab](https://www.tue.nl/en/research/research-groups/signal-processing-systems/biomedical-diagnostics-lab/) at Eindhoven University of Technology. Currently for [internal](LICENSE) use only.

In case of any questions, feel free to [contact](mailto:t.s.w.stevens@tue.nl).

Currently usbmd offers:

- Complete ultrasound signal processing and image reconstruction [pipeline](usbmd/ops.py).
- A collection of [models](usbmd/models) for ultrasound image and signal processing.
- Multi-Backend Support via [Keras3](https://keras.io/keras_3/): You can use [PyTorch](https://github.com/pytorch/pytorch), [TensorFlow](https://github.com/tensorflow/tensorflow), or [JAX](https://github.com/google/jax)


