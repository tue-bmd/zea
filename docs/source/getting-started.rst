Getting Started
===============

Installation
------------

A simple pip command will install the latest version of ``zea`` from `PyPI <https://pypi.org/project/zea>`_. Furthermore, you also have to install a backend of choice (``jax``, ``tensorflow`` or ``torch``). For more installation instructions, please refer to the :doc:`installation` page.

.. code-block:: shell

   pip install zea

Example usage
--------------

.. tip::

   A more complete set of examples can be found on the :doc:`examples` page.

``zea`` is written in Python on top of `Keras 3 <https://keras.io/about/>`_. This means that under the hood we use the Keras framework to implement the pipeline and models. Keras allows you to set a backend, which means you can use ``zea`` alongside all your projects that are implemented in their respective frameworks. To get started you first have to specify your preferred backend. This can be done by setting the ``KERAS_BACKEND`` environment variable, either in your code or in your terminal.

.. code-block:: shell

   # set the backend in your terminal
   export KERAS_BACKEND="jax"

.. code-block:: python

   # or set the backend in your code at the top of your script
   import os
   os.environ["KERAS_BACKEND"] = "jax"

After setting the backend you can simply import ``zea``

.. code-block:: python

   import zea

The easiest way to get started is to use the Interface class

.. code-block:: python

   import matplotlib.pyplot as plt

   from zea import Interface, setup

   # choose your config file
   # all necessary settings should be in the config file
   config_path = "configs/config_picmus_rf.yaml"

   # setup function handles local data paths, default config settings and GPU usage
   # make sure to create your own users.yaml using zea/datapaths.py
   users_paths = "users.yaml"
   config = setup(config_path, users_paths, create_user=True)

   # initialize the Interface class with your config
   interface = Interface(config, validate_file=False)
   image = interface.run(plot=True)

Loading a single file
~~~~~~~~~~~~~~~~~~~~~

The ``Interface`` class is a convenient way to load and inspect your data. However for more custom use cases, you might want to load and process the data yourself.
We do this by manually loading a single zea file with ``load_file`` and processing it with the ``Process`` class.

.. code-block:: python

   import keras
   import matplotlib.pyplot as plt

   from zea import setup, load_file, Pipeline

   # choose your config file
   # all necessary settings should be in the config file
   config_path = "configs/config_picmus_rf.yaml"

   # setup function handles local data paths, default config settings and GPU usage
   # make sure to create your own users.yaml using zea/datapaths.py
   users_paths = "users.yaml"
   config = setup(config_path, users_paths, create_user=True)

   # we now manually point to our data
   data_root = config.data.user.data_root
   user = config.data.user.username

   print(f"\n🔔 Hi {user}! You are using data from {data_root}\n")

   data_path = data_root / "zea_datasets/PICMUS/database/simulation/contrast_speckle/contrast_speckle_simu_dataset_rf/contrast_speckle_simu_dataset_rf.hdf5"

   # only 1 frame in PICMUS to be selected
   selected_frames = [0]

   # loading a file manually using `load_file`
   data, scan, probe = load_file(
       data_path, frames=selected_frames, scan=config.scan, data_type="raw_data"
   )

   pipeline = Pipeline.from_default(with_batch_dim=False)
   parameters = pipeline.prepare_parameters(probe, scan, config)

   # index the first frame
   data_frame = data[0]

   # processing the data from raw_data to image
   output = pipeline(data=data_frame, **parameters)
   # the output is a dictionary with all paramaters and data
   image = output["data"]
   image = keras.ops.convert_to_numpy(image)

   plt.figure()
   plt.imshow(image, cmap="gray")

   # we can also process a single plane wave angle by
   # setting the `selected_transmits` parameter in the scan object
   scan.set_transmits(1)
   parameters = pipeline.prepare_parameters(probe, scan, config)

   image = pipeline(data=data_frame, **parameters)["data"]
   image = keras.ops.convert_to_numpy(image)

   plt.figure()
   plt.imshow(image, cmap="gray")

Custom pipeline
~~~~~~~~~~~~~~~

Custom pipelines are also supported in various ways. One way is to define a pipeline in a dictionary format. Pipelines can be nested, and operations can be referenced in a list by using just their name, or by using a dictionary with the name and parameters.

.. code-block:: python

   import keras
   from zea import Config, Pipeline

   config = Config(
       {
           # operations should be a list
           "operations": [
               # operations can be just referenced by their name
               "demodulate",
               # or by name and (static) parameters
               {"name": "downsample", "params": {"factor": 4}},
               # or we can have nested pipelines even
               {
                   "name": "patched_grid",
                   "params": {
                       "operations": [
                           "tof_correction",
                           "delay_and_sum",
                       ],
                   },
               },
               "envelope_detect",
               "normalize",
               "log_compress",
           ],
       }
   )

   pipeline = Pipeline.from_config(config, with_batch_dim=False)
   parameters = pipeline.prepare_parameters(probe, scan, config)
   image = pipeline(data=data_frame, **parameters)["data"]
   image = keras.ops.convert_to_numpy(image)

   plt.figure()
   plt.imshow(image, cmap="gray")

   # change dynamic range
   image = pipeline(data=data_frame, **parameters, dynamic_range=(-30, 0))["data"]
   image = keras.ops.convert_to_numpy(image)

   plt.figure()
   plt.imshow(image, cmap="gray")


Handling multiple files (i.e. datasets)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also make use of the ``Dataset`` class to load and process multiple files at once.
We will have to manually initialize the ``Scan`` and ``Probe`` classes and pass them to the ``Process`` class. This was done automatically in the ``Interface`` in the first example.

.. code-block:: python

   import keras
   import matplotlib.pyplot as plt

   from zea import Dataset, Pipeline, init_device, setup

   device = init_device()

   # choose your config file with all your settings
   config_path = "configs/config_picmus_rf.yaml"

   # setup function handles local data paths, default config settings and GPU usage
   # make sure to create your own users.yaml using zea/datapaths.py
   users_paths = "users.yaml"
   config = setup(config_path, users_paths, create_user=True)

   # initialize the dataset
   dataset = Dataset.from_config(**config.data)

   # get the first file in the dataset and the scan and probe
   file = dataset[0]
   scan = file.scan(**config.scan)
   probe = file.probe()

   # load the data (all frames, but for picmus only one frame is available)
   data = file.load_data(dtype=config.data.dtype, indices="all")

   # initiate a pipeline (now with batch processing)
   pipeline = Pipeline.from_default()
   parameters = pipeline.prepare_parameters(probe, scan, config)
   image = pipeline(data=data, **parameters)["data"]

   # take the first frame and plot it
   plt.figure()
   plt.imshow(image[0], cmap="gray")

Models
------

``zea`` also contains a collection of models that can be used for various tasks. An example of how to use the :class:`zea.models.echonet.EchoNetDynamic` model is shown below. Simply use the :meth:`from_preset` method to load a model with a specific preset. All models can be found in the :mod:`zea.models` module. See the :doc:`models` documentation for more information.

.. code-block:: python

   import os

   # NOTE: should be `tensorflow` for EchoNetDynamic
   os.environ["KERAS_BACKEND"] = "tensorflow"

   from keras import ops
   import matplotlib.pyplot as plt

   from zea import init_device, log, set_data_paths
   from zea.models.echonet import EchoNetDynamic
   from zea.tools.selection_tool import add_shape_from_mask
   from zea.visualize import plot_image_grid, set_mpl_style
   from zea.backend.tensorflow.dataloader import make_dataloader


   data_paths = set_data_paths()
   init_device()

   val_dataset = make_dataloader(
       data_paths.data_root / "zea_datasets/CAMUS/val",
       key="data/image",
       batch_size=16,
       shuffle=True,
       image_size=[256, 256],
       resize_type="resize",
       image_range=[-60, 0],
       normalization_range=[-1, 1],
       seed=42,
   )

   presets = list(EchoNetDynamic.presets.keys())
   log.info(f"Available built-in zea presets for EchoNet: {presets}")

   model = EchoNetDynamic.from_preset("echonet-dynamic")

   batch = next(iter(val_dataset))

   masks = model(batch)

   masks = ops.squeeze(masks, axis=-1)
   masks = ops.convert_to_numpy(masks)

   set_mpl_style()

   # create figure of images in batch
   fig, _ = plot_image_grid(batch)
   axes = fig.axes[:batch.shape[0]]
   for ax, mask in zip(axes, masks):
       # add segmentation on top of image in figure
       add_shape_from_mask(ax, mask, color="red", alpha=0.5)
   plt.show()
